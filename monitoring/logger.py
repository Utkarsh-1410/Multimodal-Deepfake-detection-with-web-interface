"""
Enhanced Logging and Monitoring System
Provides structured JSON logging, metrics tracking, and performance monitoring
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
from collections import defaultdict


class StructuredLogger:
    """
    Structured JSON logger for deepfake detection pipeline.
    Logs all operations, metrics, and errors in JSON format for easy parsing.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """
        Args:
            log_dir: Directory to save log files
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"detection_{timestamp}.jsonl"
        
        # Setup standard Python logger
        self.logger = logging.getLogger("DeepFakeDetection")
        self.logger.setLevel(log_level)
        
        # File handler for JSON logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.session_start = time.time()
    
    def log_event(self, event_type: str, data: Dict[str, Any], level: str = "INFO"):
        """
        Log a structured event.
        
        Args:
            event_type: Type of event (e.g., 'video_processed', 'model_loaded')
            data: Event data dictionary
            level: Log level
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'level': level,
            'data': data
        }
        
        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also log to standard logger
        message = f"{event_type}: {json.dumps(data)}"
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def log_video_processing(self, video_path: str, method: str, 
                            result: Dict[str, Any], processing_time: float):
        """Log video processing result"""
        self.log_event(
            'video_processed',
            {
                'video_path': video_path,
                'method': method,
                'result': result,
                'processing_time_seconds': processing_time
            }
        )
    
    def log_model_loaded(self, model_name: str, model_path: str, load_time: float):
        """Log model loading"""
        self.log_event(
            'model_loaded',
            {
                'model_name': model_name,
                'model_path': model_path,
                'load_time_seconds': load_time
            }
        )
    
    def log_training_epoch(self, epoch: int, metrics: Dict[str, float], 
                         fold: Optional[int] = None):
        """Log training epoch metrics"""
        data = {
            'epoch': epoch,
            'metrics': metrics
        }
        if fold is not None:
            data['fold'] = fold
        
        self.log_event('training_epoch', data)
        
        # Store metrics
        for key, value in metrics.items():
            self.metrics[f'epoch_{key}'].append(value)
    
    def log_error(self, error_type: str, error_message: str, 
                 context: Optional[Dict[str, Any]] = None):
        """Log error"""
        data = {
            'error_type': error_type,
            'error_message': error_message
        }
        if context:
            data['context'] = context
        
        self.log_event('error', data, level='ERROR')
    
    def log_metrics(self, method: str, metrics: Dict[str, float]):
        """Log evaluation metrics"""
        self.log_event(
            'evaluation_metrics',
            {
                'method': method,
                'metrics': metrics
            }
        )
        
        # Store metrics
        for key, value in metrics.items():
            self.metrics[f'{method}_{key}'].append(value)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        session_time = time.time() - self.session_start
        
        summary = {
            'session_duration_seconds': session_time,
            'log_file': str(self.log_file),
            'metrics_summary': {}
        }
        
        # Compute averages for stored metrics
        for key, values in self.metrics.items():
            if values:
                summary['metrics_summary'][key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return summary
    
    def save_summary(self, output_path: Optional[str] = None):
        """Save session summary to file"""
        summary = self.get_session_summary()
        
        if output_path is None:
            output_path = self.log_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return output_path


class MetricsCollector:
    """
    Collects and aggregates metrics across multiple runs.
    """
    
    def __init__(self):
        self.collected_metrics = defaultdict(list)
        self.timestamps = []
    
    def add_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Add a metric value"""
        if timestamp is None:
            timestamp = time.time()
        
        self.collected_metrics[name].append(value)
        self.timestamps.append(timestamp)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.collected_metrics:
            return {}
        
        values = self.collected_metrics[name]
        if not values:
            return {}
        
        return {
            'mean': sum(values) / len(values),
            'std': self._std(values),
            'min': min(values),
            'max': max(values),
            'median': sorted(values)[len(values) // 2],
            'count': len(values)
        }
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def export_to_json(self, output_path: str):
        """Export all metrics to JSON"""
        data = {
            'metrics': dict(self.collected_metrics),
            'statistics': {
                name: self.get_statistics(name)
                for name in self.collected_metrics.keys()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def reset(self):
        """Reset all collected metrics"""
        self.collected_metrics.clear()
        self.timestamps.clear()

