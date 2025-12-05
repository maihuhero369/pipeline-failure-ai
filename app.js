// app.js — main UI logic: robust parse, cleaning, cpu backend, training with logs
(async function(){

  // DOM
  const fileInput = document.getElementById('fileInput');
  const parseBtn = document.getElementById('parseBtn');
  const loadExampleBtn = document.getElementById('loadExampleBtn');

  const trainBtn = document.getElementById('trainBtn');
  const saveBtn = document.getElementById('saveModelBtn');
  const loadBtn = document.getElementById('loadModelBtn');

  const predictBtn = document.getElementById('predictBtn');
  const resetBtn = document.getElementById('resetModelBtn');

  const predictInputs = document.getElementById('predictInputs');
  const predictResult = document.getElementById('predictResult');
  const dataSummary = document.getElementById('dataSummary');
  const featureStats = document.getElementById('featureStats');
  const sampleTable = document.getElementById('sampleTable');
  const trainLogs = document.getElementById('trainLogs');

  const openDocsBtn = document.getElementById('openDocsBtn');
  const closeDocsBtn = document.getElementById('closeDocsBtn');
  const docsOverlay = document.getElementById('docsOverlay');
  const showSampleBtn = document.getElementById('showSampleBtn');

  // Ensure TF backend stable: use CPU for maximum compatibility
  try{
    await tf.setBackend('cpu');
    Utils.log('TF backend set to: ' + tf.getBackend());
  }catch(e){ Utils.log('TF backend set failed — proceeding with default: ' + (tf.getBackend ? tf.getBackend() : 'unknown')); }

  // Chart holders
  let chartLoss = null;

  // parsed data holder
  let parsed = null;
  let featureNames = [];

  // UTIL: display sample rows
  function displaySampleTable(headers, rows){
    const n = Math.min(8, rows.length);
    if(n===0){ sampleTable.innerHTML = '<i>No rows parsed yet</i>'; return; }
    let html = '<table><thead><tr>';
    headers.forEach(h=> html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';
    for(let i=0;i<n;i++){
      html += '<tr>';
      headers.forEach(h=> html += `<td>${rows[i][h] ?? ''}</td>`);
      html += '</tr>';
    }
    html += '</tbody></table>';
    sampleTable.innerHTML = html;
  }

  function computeStats(rows, features){
    const stats = {};
    features.forEach(f=>{
      const vals = rows.map(r=> Number(r[f])).filter(v=> Number.isFinite(v) && !Number.isNaN(v));
      if(vals.length === 0){ stats[f] = null; return; }
      const min = Math.min(...vals), max = Math.max(...vals);
      const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
      stats[f] = {min, max, mean};
    });
    return stats;
  }

  function showDataSummary(headers, rows){
    dataSummary.innerHTML = `<strong>Columns:</strong> ${headers.join(', ')} • <strong>Rows:</strong> ${rows.length}`;
    const stats = computeStats(rows, headers.filter(h=>h!=='failure'));
    let shtml = '<strong>Feature stats:</strong><br>';
    Object.keys(stats).forEach(k=>{
      const st = stats[k];
      shtml += st ? `<div>${k}: min=${st.min.toFixed(3)} max=${st.max.toFixed(3)} mean=${st.mean.toFixed(3)}</div>` : `<div>${k}: <em>non-numeric</em></div>`;
    });
    featureStats.innerHTML = shtml;
    displaySampleTable(headers, rows);
  }

  function setupPredictInputs(features, stats){
    predictInputs.innerHTML = '';
    features.forEach(f=>{
      const st = stats[f];
      const placeholder = st ? `${st.min.toFixed(2)} – ${st.max.toFixed(2)}` : '';
      const div = document.createElement('div');
      div.innerHTML = `<label>${f}: <input data-feature="${f}" type="number" step="any" placeholder="${placeholder}" /></label>`;
      predictInputs.appendChild(div);
    });
  }

  // Chart helpers
  function setupLossChart(){
    const ctx = document.getElementById('lossChart');
    if(chartLoss) chartLoss.destroy();
    chartLoss = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Loss', data: [], borderColor:'#00c6ff', fill:false, tension:0.2 },
        { label: 'Accuracy', data: [], borderColor:'#7c3aed', fill:false, tension:0.2 }
      ]},
      options: { responsive:true, maintainAspectRatio:false, animation:false, scales:{ x:{title:{display:true,text:'Epoch'}}, y:{beginAtZero:true}} }
    });
    return chartLoss;
  }

  function updateLossChart(lossArr, accArr){
    if(!chartLoss) return;
    chartLoss.data.labels = lossArr.map(p=>p.x);
    chartLoss.data.datasets[0].data = lossArr.map(p=>p.y);
    chartLoss.data.datasets[1].data = accArr.map(p=>p.y);
    chartLoss.update();
  }

  function drawFeaturePreview(X, featureName){
    try{
      const values = X.map(r=>r[0]);
      const ctx = document.getElementById('featureChart');
      if(window._featureChart) window._featureChart.destroy();
      window._featureChart = new Chart(ctx, {
        type:'bar',
        data:{ labels: values.slice(0,30).map((_,i)=>i+1), datasets:[{ label: featureName, data: values.slice(0,30), backgroundColor:'#00c6ff55'}]},
        options:{ responsive:true, maintainAspectRatio:false }
      });
    }catch(e){ Utils.log('Feature preview error: '+e.message); }
  }

  // CSV events
  fileInput.addEventListener('change', async ()=>{
    const f = fileInput.files[0];
    if(!f) return;
    const txt = await f.text();
    parsed = Utils.parseCSVText(txt);
    Utils.log('CSV loaded — headers: ' + parsed.headers.join(', '));
    parseBtn.disabled = false;
  });

  loadExampleBtn.addEventListener('click', ()=>{
    const txt = Utils.sampleExampleCSV();
    parsed = Utils.parseCSVText(txt);
    Utils.log('Example CSV generated');
    parseBtn.disabled = false;
  });

  parseBtn.addEventListener('click', ()=>{
    if(!parsed){ alert('No CSV loaded'); return; }
    const labelName = 'failure';
    if(!parsed.headers.includes(labelName)){ alert('CSV must include "failure" column'); return; }

    // prepare features
    featureNames = parsed.headers.filter(h => h !== labelName);
    // show summary & stats
    showDataSummary(parsed.headers, parsed.rows);

    // convert to numeric and drop bad rows
    const {X, y} = Utils.toNumericMatrix(parsed.rows, featureNames, labelName);
    if(X.length === 0){ alert('No valid numeric rows found after parsing'); return; }
    const norm = Utils.normalize(X);
    window._trainData = { X: norm.Xn, y, mins: norm.mins, maxs: norm.maxs, rawX: X };

    // compute stats for placeholders using numeric rows
    const stats = computeStats(parsed.rows.filter((r,i)=>{
      // keep rows that toNumericMatrix would keep
      const vals = featureNames.map(f => Number(r[f]));
      const lab = Number(r[labelName]);
      return vals.every(v => Number.isFinite(v)) && Number.isFinite(lab);
    }), featureNames);

    setupPredictInputs(featureNames, stats);
    drawFeaturePreview(X, featureNames[0]);
    trainBtn.disabled = false;
    Utils.log('Parsed and prepared train data. Rows: ' + X.length);
  });

  // Train button
  trainBtn.addEventListener('click', async ()=>{
    if(!window._trainData){ alert('No training data — parse CSV first'); return; }
    trainBtn.disabled = true;
    try{
      Utils.log('Preparing tensors...');
      const epochs = Number(document.getElementById('epochs').value);
      const batchSize = Number(document.getElementById('batchSize').value);
      const lr = Number(document.getElementById('lr').value);
      const X = tf.tensor2d(window._trainData.X);
      const y = tf.tensor2d(window._trainData.y, [window._trainData.y.length,1]);

      ModelAPI.createModel(X.shape[1], lr);
      const lossArr = [], accArr = [];
      setupLossChart();

      Utils.log('Training started...');
      await ModelAPI.trainModel(X, y, epochs, batchSize, (epoch, logs) => {
        const acc = logs.acc ?? logs.accuracy ?? 0;
        lossArr.push({x: epoch+1, y: logs.loss});
        accArr.push({x: epoch+1, y: acc});
        updateLossChart(lossArr, accArr);
        trainLogs.innerText = `Epoch ${epoch+1} — loss: ${logs.loss.toFixed(4)} acc: ${acc.toFixed(3)}`;
        Utils.log(`Epoch ${epoch+1} — loss:${logs.loss.toFixed(4)} acc:${acc.toFixed(3)}`);
      });

      Utils.log('Training complete.');
      saveBtn.disabled = false;
      predictBtn.disabled = false;
    }catch(err){
      Utils.log('Training failed: ' + (err.message || err));
      alert('Training error — open console output for details.');
    }finally{
      trainBtn.disabled = false;
    }
  });

  // Save & load
  saveBtn.addEventListener('click', async ()=>{
    try{ await ModelAPI.saveModelToLocal(); Utils.log('Model saved to localStorage'); } catch(e){ Utils.log('Save failed: ' + e.message); alert('Save failed: ' + e.message); }
  });

  loadBtn.addEventListener('click', async ()=>{
    try{ await ModelAPI.loadModelFromLocal(); Utils.log('Model loaded from localStorage'); predictBtn.disabled = false; } catch(e){ Utils.log('Load failed: ' + e.message); alert('Load failed: ' + e.message); }
  });

  // Predict
  predictBtn.addEventListener('click', async ()=>{
    if(!window._trainData){ alert('Train or load a model first'); return; }
    const vals = Array.from(predictInputs.querySelectorAll('input')).map(i => i.value.trim() === '' ? NaN : Number(i.value));
    if(vals.some(v=>Number.isNaN(v))){ alert('Please enter all numeric feature values'); return; }
    try{
      const p = await ModelAPI.predictArray(vals, window._trainData.mins, window._trainData.maxs);
      predictResult.innerHTML = `<div style="padding:8px;background:rgba(255,255,255,0.02);border-radius:8px">Failure probability: <strong>${(p*100).toFixed(2)}%</strong></div>`;
      Utils.log('Predicted probability: ' + p.toFixed(4));
    }catch(e){
      Utils.log('Prediction error: ' + (e.message || e));
      alert('Prediction failed. Train or load a model first.');
    }
  });

  resetBtn.addEventListener('click', ()=>{ ModelAPI.resetModel(); Utils.log('Model reset'); saveBtn.disabled = true; predictBtn.disabled = true; });

  // Docs overlay
  openDocsBtn.addEventListener('click', ()=> docsOverlay.classList.remove('hidden'));
  closeDocsBtn?.addEventListener('click', ()=> docsOverlay.classList.add('hidden'));

  // Show/hide sample
  showSampleBtn.addEventListener('click', ()=> {
    if(sampleTable.style.display === 'none' || sampleTable.innerHTML.trim() === ''){
      if(parsed) displaySampleTable(parsed.headers, parsed.rows);
      else sampleTable.innerHTML = '<i>No CSV parsed yet. Click Example CSV or upload your CSV.</i>';
      sampleTable.style.display = 'block';
    } else {
      sampleTable.style.display = 'none';
    }
  });

  // init
  Utils.log('App ready. Use Example CSV to test quickly.');

  // resize charts on window resize
  window.addEventListener('resize', ()=>{ try{ if(window._featureChart) window._featureChart.resize(); if(chartLoss) chartLoss.resize(); } catch(e){} });

})();
