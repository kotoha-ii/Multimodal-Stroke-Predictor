// 录音相关变量
let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let canvasCtx;
const canvas = document.createElement('canvas');
document.getElementById('audioVisualizer').appendChild(canvas);
canvas.width = document.getElementById('audioVisualizer').offsetWidth;
canvas.height = document.getElementById('audioVisualizer').offsetHeight;
canvasCtx = canvas.getContext('2d');

// 录音控制
document.getElementById('startRecord').addEventListener('click', startRecording);
document.getElementById('stopRecord').addEventListener('click', stopRecording);
document.getElementById('uploadRecord').addEventListener('click', uploadRecording);
document.getElementById('uploadForm').addEventListener('submit', handleFileUpload);

// 开始录音
async function startRecording() {
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    
    // 设置音频分析
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 256;
    
    // 开始可视化
    visualize();
    
    mediaRecorder.addEventListener('dataavailable', event => {
        audioChunks.push(event.data);
    });
    
    mediaRecorder.start();
    toggleRecordingButtons(true);
}

// 停止录音
function stopRecording() {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    toggleRecordingButtons(false);
}

// 上传录音
function uploadRecording() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    sendAudioForAnalysis(audioBlob);
}

// 处理文件上传
function handleFileUpload(e) {
    e.preventDefault();
    const file = document.getElementById('audioFile').files[0];
    if (file) {
        sendAudioForAnalysis(file);
    }
}

// 发送音频到后端分析
function sendAudioForAnalysis(audioData) {
    const formData = new FormData();
    formData.append('audio', audioData, 'recording.wav');
    
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(displayResults)
    .catch(error => {
        console.error('Error:', error);
        alert('分析失败: ' + error.message);
    });
}

// 显示结果
function displayResults(data) {
    const resultCard = document.getElementById('resultCard');
    const resultsDiv = document.getElementById('results');
    
    // 清空之前的结果
    resultsDiv.innerHTML = '';
    
    // 显示基本信息
    const summary = document.createElement('p');
    summary.textContent = data.summary;
    resultsDiv.appendChild(summary);
    
    // 显示声学特征
    const acousticTitle = document.createElement('h6');
    acousticTitle.textContent = '声学特征分析';
    resultsDiv.appendChild(acousticTitle);
    
    const acousticList = document.createElement('ul');
    for (const [key, value] of Object.entries(data.acoustic_features)) {
        const item = document.createElement('li');
        item.textContent = `${key}: ${value}`;
        acousticList.appendChild(item);
    }
    resultsDiv.appendChild(acousticList);
    
    // 显示语言学特征
    const linguisticTitle = document.createElement('h6');
    linguisticTitle.textContent = '语言学特征分析';
    resultsDiv.appendChild(linguisticTitle);
    
    const linguisticList = document.createElement('ul');
    for (const [key, value] of Object.entries(data.linguistic_features)) {
        const item = document.createElement('li');
        item.textContent = `${key}: ${value}`;
        linguisticList.appendChild(item);
    }
    resultsDiv.appendChild(linguisticList);
    
    // 显示医学指标
    const medicalTitle = document.createElement('h6');
    medicalTitle.textContent = '医学指标评估';
    resultsDiv.appendChild(medicalTitle);
    
    const medicalList = document.createElement('ul');
    for (const [key, value] of Object.entries(data.medical_indicators)) {
        const item = document.createElement('li');
        item.textContent = `${key}: ${value}`;
        medicalList.appendChild(item);
    }
    resultsDiv.appendChild(medicalList);
    
    // 显示结果卡片
    resultCard.style.display = 'block';
}

// 音频可视化
function visualize() {
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    function draw() {
        requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(dataArray);
        
        canvasCtx.fillStyle = 'rgb(233, 236, 239)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'rgb(13, 110, 253)';
        canvasCtx.beginPath();
        
        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;
        
        for(let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            
            if(i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        canvasCtx.lineTo(canvas.width, canvas.height/2);
        canvasCtx.stroke();
    }
    
    draw();
}

// 切换录音按钮状态
function toggleRecordingButtons(isRecording) {
    document.getElementById('startRecord').disabled = isRecording;
    document.getElementById('stopRecord').disabled = !isRecording;
    document.getElementById('uploadRecord').disabled = !isRecording || audioChunks.length === 0;
}