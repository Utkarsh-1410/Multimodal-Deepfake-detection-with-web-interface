"""
Job Orchestration System for Batch Video Processing
Provides queue-based processing with progress tracking and error recovery
"""

import os
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import threading
from queue import Queue, Empty
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a single processing job"""
    job_id: str
    video_path: str
    method: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create from dictionary"""
        data['status'] = JobStatus(data['status'])
        return cls(**data)


class JobQueue:
    """
    Queue-based job orchestrator for batch video processing.
    Supports parallel processing, retries, and progress tracking.
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 save_state: bool = True,
                 state_file: str = "job_queue_state.json"):
        """
        Args:
            max_workers: Maximum number of parallel workers
            save_state: Whether to save job state to disk
            state_file: Path to state file
        """
        self.max_workers = max_workers
        self.save_state = save_state
        self.state_file = Path(state_file)
        
        self.jobs: Dict[str, Job] = {}
        self.queue: Queue = Queue()
        self.lock = threading.Lock()
        
        # Load existing state if available
        if self.save_state and self.state_file.exists():
            self.load_state()
    
    def add_job(self, video_path: str, method: str, 
               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a job to the queue.
        
        Args:
            video_path: Path to video file
            method: Detection method
            metadata: Optional metadata
        
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            video_path=video_path,
            method=method,
            status=JobStatus.PENDING,
            created_at=datetime.now().isoformat(),
            metadata=metadata
        )
        
        with self.lock:
            self.jobs[job_id] = job
            self.queue.put(job_id)
        
        if self.save_state:
            self.save_state()
        
        return job_id
    
    def add_jobs_batch(self, video_paths: List[str], method: str,
                      metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add multiple jobs to the queue"""
        job_ids = []
        for i, video_path in enumerate(video_paths):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            job_id = self.add_job(video_path, method, metadata)
            job_ids.append(job_id)
        return job_ids
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status"""
        job = self.get_job(job_id)
        return job.status if job else None
    
    def update_job(self, job_id: str, **kwargs):
        """Update job fields"""
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
        
        if self.save_state:
            self.save_state()
    
    def process_jobs(self, 
                    process_func: Callable[[str, str], Dict[str, Any]],
                    progress_callback: Optional[Callable[[Job], None]] = None):
        """
        Process all jobs in the queue.
        
        Args:
            process_func: Function to process a single job (video_path, method) -> result
            progress_callback: Optional callback for progress updates
        """
        def worker(job_id: str):
            """Worker function to process a single job"""
            job = self.get_job(job_id)
            if not job:
                return
            
            # Update status to running
            self.update_job(job_id, 
                          status=JobStatus.RUNNING,
                          started_at=datetime.now().isoformat())
            
            try:
                # Process the job
                result = process_func(job.video_path, job.method)
                
                # Update with result
                self.update_job(job_id,
                              status=JobStatus.COMPLETED,
                              completed_at=datetime.now().isoformat(),
                              result=result)
                
                if progress_callback:
                    progress_callback(job)
                
            except Exception as e:
                # Handle failure
                job.retry_count += 1
                
                if job.retry_count < job.max_retries:
                    # Retry
                    self.update_job(job_id,
                                  status=JobStatus.PENDING,
                                  error=str(e),
                                  retry_count=job.retry_count)
                    self.queue.put(job_id)  # Re-queue for retry
                else:
                    # Max retries reached
                    self.update_job(job_id,
                                  status=JobStatus.FAILED,
                                  completed_at=datetime.now().isoformat(),
                                  error=str(e))
                
                if progress_callback:
                    progress_callback(job)
        
        # Process jobs with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Submit initial batch
            while not self.queue.empty():
                try:
                    job_id = self.queue.get_nowait()
                    future = executor.submit(worker, job_id)
                    futures[future] = job_id
                except Empty:
                    break
            
            # Process completed futures and submit new ones
            while futures:
                done, not_done = as_completed(futures, timeout=1), []
                for future in done:
                    job_id = futures.pop(future)
                    try:
                        future.result()  # Check for exceptions
                    except Exception as e:
                        print(f"Error processing job {job_id}: {e}")
                    
                    # Submit next job if available
                    try:
                        job_id = self.queue.get_nowait()
                        future = executor.submit(worker, job_id)
                        futures[future] = job_id
                    except Empty:
                        pass
    
    def get_progress(self) -> Dict[str, Any]:
        """Get overall progress statistics"""
        with self.lock:
            total = len(self.jobs)
            if total == 0:
                return {'total': 0}
            
            status_counts = {}
            for status in JobStatus:
                status_counts[status.value] = sum(
                    1 for job in self.jobs.values() if job.status == status
                )
            
            return {
                'total': total,
                'status_counts': status_counts,
                'completed': status_counts.get('completed', 0),
                'failed': status_counts.get('failed', 0),
                'running': status_counts.get('running', 0),
                'pending': status_counts.get('pending', 0),
                'progress_percent': (status_counts.get('completed', 0) / total * 100) if total > 0 else 0
            }
    
    def save_state(self):
        """Save job state to disk"""
        with self.lock:
            state = {
                'jobs': [job.to_dict() for job in self.jobs.values()],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load job state from disk"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            with self.lock:
                self.jobs = {}
                for job_data in state.get('jobs', []):
                    job = Job.from_dict(job_data)
                    self.jobs[job.job_id] = job
                    
                    # Re-queue pending jobs
                    if job.status == JobStatus.PENDING:
                        self.queue.put(job.job_id)
            
            print(f"Loaded {len(self.jobs)} jobs from state file")
        except Exception as e:
            print(f"Error loading state: {e}")
    
    def cancel_job(self, job_id: str):
        """Cancel a job"""
        self.update_job(job_id, status=JobStatus.CANCELLED)
    
    def get_failed_jobs(self) -> List[Job]:
        """Get all failed jobs"""
        with self.lock:
            return [job for job in self.jobs.values() if job.status == JobStatus.FAILED]
    
    def retry_failed_jobs(self):
        """Retry all failed jobs"""
        failed_jobs = self.get_failed_jobs()
        for job in failed_jobs:
            job.retry_count = 0
            job.status = JobStatus.PENDING
            job.error = None
            self.queue.put(job.job_id)
        
        if self.save_state:
            self.save_state()

