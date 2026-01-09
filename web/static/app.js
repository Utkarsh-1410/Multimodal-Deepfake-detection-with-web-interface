const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze');
const methodSelect = document.getElementById('method');
const loadingEl = document.getElementById('loading');
const resultEl = document.getElementById('result');
const errorEl = document.getElementById('error');
const progressSection = document.getElementById('progress-section');
const progressBarInner = document.getElementById('progress-bar-inner');
const stepEls = document.querySelectorAll('.progress-steps .step');
const progressDesc = document.getElementById('progress-desc');

let selectedFile = null;

const DETECTION_STEPS = [
  { key: 'uploading', label: 'Uploading', desc: 'Uploading video...' },
  { key: 'extracting', label: 'Extracting faces', desc: 'Extracting faces from video...' },
  { key: 'mri', label: 'Generating MRIs', desc: 'Generating MRI perceptual maps...' },
  { key: 'detection', label: 'Running detection', desc: 'Classifying video as REAL or DEEP-FAKE...' },
  { key: 'done', label: 'Done', desc: 'Done!' }
];

let progressStepIdx = 0;
let progressInterval = null;
function showProgress(stepsToShow = DETECTION_STEPS) {
  progressSection.classList.remove('hidden');
  stepEls.forEach((el, i) => {
    el.classList.remove('active','done');
    if (i < progressStepIdx) el.classList.add('done');
    else if (i === progressStepIdx) el.classList.add('active');
  });
  const percent = Math.min(100, (progressStepIdx/(stepsToShow.length-1))*100);
  progressBarInner.style.width = percent+"%";
  progressDesc.textContent = stepsToShow[progressStepIdx]?.desc || '';
}
function hideProgress() {
  progressSection.classList.add('hidden');
  progressBarInner.style.width = "0%";
  stepEls.forEach(el=>el.classList.remove('active','done'));
}

function setLoading(isLoading, steps=DETECTION_STEPS) {
  if (isLoading) {
    hideProgress();
    progressStepIdx = 0;
    showProgress(steps);
    progressSection.classList.remove('hidden');
    analyzeBtn.disabled = true;
  } else {
    hideProgress();
    analyzeBtn.disabled = !selectedFile;
  }
  loadingEl.classList.toggle('hidden', isLoading);
}

function displayMethod(method) {
  if (method === 'plain_frames') return 'mri_gan';
  return method;
}

function setResult(label, probability, method) {
  loadingEl.classList.add('hidden');
  progressSection.classList.add('hidden');
  resultEl.classList.remove('hidden');
  errorEl.classList.add('hidden');
  const pillClass = label === 'REAL' ? 'real' : 'fake';
  resultEl.innerHTML = `
    <div class="pill ${pillClass}">${label} · ${probability}%</div>
    <div class="muted" style="margin-top:8px">Method: ${displayMethod(method)}</div>
  `;
}

function setError(msg) {
  loadingEl.classList.add('hidden');
  progressSection.classList.add('hidden');
  errorEl.classList.remove('hidden');
  resultEl.classList.add('hidden');
  errorEl.textContent = msg;
}

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  analyzeBtn.disabled = false;
  dropzone.classList.remove('dragover');
  dropzone.querySelector('.icon').textContent = '✅';
}

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

// Simulated progress worker for frontend only
async function simulateDetectionProgress(selectedMethod) {
  let steps = [...DETECTION_STEPS];
  // MRI/fusion methods: show MRI step; other methods: skip
  if (selectedMethod !== 'MRI' && selectedMethod !== 'fusion') {
    steps = steps.filter(s=>s.key!=='mri');
  }
  setLoading(true, steps);
  for(let i=0;i<steps.length;i++){
    progressStepIdx = i;
    showProgress(steps);
    // Animate progress, step duration: upload=400ms, extract=1200ms, mri=1500ms, detect=2000ms, done=300ms
    let dur = 800;
    if(steps[i].key==='uploading') dur=400;
    else if(steps[i].key==='extracting') dur=1000;
    else if(steps[i].key==='mri') dur=1500;
    else if(steps[i].key==='detection') dur=1600;
    else dur=360;
    await new Promise(r=>setTimeout(r, dur));
  }
  progressStepIdx = steps.length-1;
  showProgress(steps);
  setTimeout(()=>{
    setLoading(false, steps);
  },350);
}

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  errorEl.classList.add('hidden');
  resultEl.classList.add('hidden');
  const method = methodSelect.value;
  simulateDetectionProgress(method);// run simulated progress in parallel
  // For smoothness, "upload" step is before API. We'll call API after a brief delay
  await new Promise(r=>setTimeout(r, 550));
  const form = new FormData();
  form.append('file', selectedFile);
  form.append('method', method);
  try {
    const res = await fetch('/api/detect', {
      method: 'POST',
      body: form
    });
    const data = await res.json();
    if (!res.ok || data.status !== 'ok') {
      throw new Error(data.detail || 'Detection failed');
    }
    const { label, probability_percent, method } = data.result;
    setTimeout(()=>setResult(label, probability_percent, method),500);
  } catch (err) {
    setTimeout(()=>setError(err.message || 'Something went wrong'),500);
  } finally {
    // done step hides progress bar shortly after
    setTimeout(()=>setLoading(false),1200);
  }
});


