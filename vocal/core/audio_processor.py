import librosa
import numpy as np
from pyAudioAnalysis.audioBasicIO import read_audio_file
from pyAudioAnalysis import ShortTermFeatures
import speech_recognition as sr
from pypinyin import pinyin, Style
import jieba
import speech_recognition as sr
from pypinyin import pinyin, Style
import jieba
# 简化特征提取，主要使用LibROSA

def analyze_audio(filepath):
    """分析音频文件并返回脑卒中相关特征"""
    try:
        # 加载音频文件
        y, sr = librosa.load(filepath, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 语音识别
        text = speech_to_text(filepath)
        
        # 提取声学特征
        features = {
            'basic': extract_basic_features(y, sr),
            'advanced': extract_advanced_features(filepath, sr),
            'linguistic': extract_linguistic_features(y, sr)
        }
        
        # 如果语音识别成功，添加音素分析
        if text:
            features['phonetic'] = analyze_phonetic_accuracy(text)
        
        # 生成分析报告
        report = generate_report(features, duration)
        return report
        
    except Exception as e:
        print(f"音频分析错误: {str(e)}")
        return {
            'error': f"音频分析失败: {str(e)}",
            'summary': "无法处理音频文件"
        }

def speech_to_text(filepath):
    """将语音转换为文字"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filepath) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language='zh-CN')
            print(f"识别文本: {text}")  # 添加调试输出
            return text
    except Exception as e:
        print(f"语音识别错误: {str(e)}")
        return None

def analyze_phonetic_accuracy(text):
    """分析元音和辅音的准确率"""
    if not text:
        return None
        
    # 中文声母韵母对照表
    vowels = set('aeiouüāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ')
    consonants = set('bpmfdtnlgkhjqxzhchshrzcs')
    
    # 获取拼音
    pinyin_list = [item[0] for item in pinyin(text, style=Style.TONE)]
    
    # 分析元音和辅音
    vowel_count = 0
    vowel_total = 0
    consonant_count = 0
    consonant_total = 0
    
    for py in pinyin_list:
        # 分析每个拼音中的元音
        for char in py:
            if char in vowels:
                vowel_count += 1
                vowel_total += 1
        
        # 分析声母（辅音）
        if py[0] in consonants:
            consonant_count += 1
            consonant_total += 1
    
    return {
        'vowel_accuracy': vowel_count / vowel_total if vowel_total > 0 else 0,
        'consonant_accuracy': consonant_count / consonant_total if consonant_total > 0 else 0,
        'text': text,
        'pinyin': pinyin_list
    }

def speech_to_text(filepath):
    """将语音转换为文字"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filepath) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language='zh-CN')
            return text
    except Exception as e:
        print(f"语音识别错误: {str(e)}")
        print("继续使用基础声学特征进行分析")
        return None

def analyze_phonetic_accuracy(text):
    """分析元音和辅音的准确率"""
    if not text:
        return None
        
    # 中文声母韵母对照表
    vowels = set('aeiouüāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ')
    consonants = set('bpmfdtnlgkhjqxzhchshrzcs')
    
    # 获取拼音
    pinyin_list = [item[0] for item in pinyin(text, style=Style.TONE)]
    
    # 分析元音和辅音
    vowel_count = 0
    vowel_total = 0
    consonant_count = 0
    consonant_total = 0
    
    for py in pinyin_list:
        # 分析每个拼音中的元音
        for char in py:
            if char in vowels:
                vowel_count += 1
                vowel_total += 1
        
        # 分析声母（辅音）
        if py[0] in consonants:
            consonant_count += 1
            consonant_total += 1
    
    return {
        'vowel_accuracy': vowel_count / vowel_total if vowel_total > 0 else 0,
        'consonant_accuracy': consonant_count / consonant_total if consonant_total > 0 else 0,
        'text': text,
        'pinyin': pinyin_list
    }

def extract_basic_features(y, sr):
    """提取基础声学特征"""
    # MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # 基频特征
    f0 = librosa.yin(y, fmin=50, fmax=500)
    f0 = f0[f0 > 0]  # 去除无效值
    f0_mean = np.mean(f0) if len(f0) > 0 else 0
    f0_std = np.std(f0) if len(f0) > 0 else 0
    
    return {
        'mfcc': mfcc_mean.tolist(),
        'f0_mean': float(f0_mean),
        'f0_std': float(f0_std)
    }

  # 替代旧模块

def extract_advanced_features(filepath, sr):
    """使用新版 pyAudioAnalysis 提取高级特征"""
    [Fs, x] = read_audio_file(filepath)
    if Fs != sr:
        x = librosa.resample(x, orig_sr=Fs, target_sr=sr)
        Fs = sr
    
    # 新版特征提取接口
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    
    # 动态获取特征索引（新版特征名称需核对）
    jitter_idx = f_names.index('jitter') if 'jitter' in f_names else None
    shimmer_idx = f_names.index('shimmer') if 'shimmer' in f_names else None
    harmonic_idx = f_names.index('harmonic_ratio') if 'harmonic_ratio' in f_names else None
    
    return {
        'jitter': float(np.mean(F[jitter_idx, :])) if jitter_idx else 0.0,
        'shimmer': float(np.mean(F[shimmer_idx, :])) if shimmer_idx else 0.0,
        'harmonic_ratio': float(np.mean(F[harmonic_idx, :])) if harmonic_idx else 0.0
    }

def extract_linguistic_features(y, sr):
    """提取语言学特征"""
    # 计算语速（音节/秒）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    # 改进的停顿检测
    # 1. 使用自适应阈值
    rms = librosa.feature.rms(y=y)[0]
    silence_threshold = np.mean(rms) * 0.1  # 使用均值的10%作为基准
    
    # 2. 使用最小持续时间
    frame_length = int(0.1 * sr)  # 100ms最小静音段
    hop_length = int(0.05 * sr)   # 50ms步长
    
    # 3. 计算帧级别的能量
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames**2, axis=0))
    
    # 4. 标记静音段
    is_silence = frame_rms < silence_threshold
    
    # 5. 应用最小持续时间约束
    min_silence_frames = 3  # 至少3帧（150ms）才算真正的停顿
    silence_runs = np.split(is_silence, np.where(np.diff(is_silence))[0] + 1)
    valid_silences = [run for run in silence_runs if len(run) >= min_silence_frames and run[0]]
    
    # 计算停顿频率
    pause_count = len(valid_silences)
    duration = len(y) / sr
    pause_frequency = pause_count / duration if duration > 0 else 0
    
    return {
        'speech_rate': float(tempo/60),  # 转换为音节/秒
        'pause_frequency': float(pause_frequency)
    }

def generate_report(features, duration):
    """生成中文分析报告"""
    basic = features['basic']
    advanced = features['advanced']
    linguistic = features['linguistic']
    
    # 构音障碍风险评估
    articulation_score = 0
    # 1. 基频稳定性评估 (权重: 0.4)
    f0_std_score = 1 - min(basic['f0_std'] / 100, 1)  # 标准差越大，分数越低
    articulation_score += 0.4 * f0_std_score
    
    # 2. 谐波比率评估 (权重: 0.4)
    harmonic_score = min(advanced['harmonic_ratio'], 1)  # 谐波比率越高，分数越高
    articulation_score += 0.4 * harmonic_score
    
    # 3. 音高变化评估 (权重: 0.2)
    jitter_score = 1 - min(advanced['jitter'] * 100, 1)  # jitter越小，分数越高
    articulation_score += 0.2 * jitter_score
    
    # 失语症风险评估
    aphasia_score = 0
    # 1. 语速评估 (权重: 0.4)
    normal_speech_rate = 4.0  # 正常语速约4音节/秒
    rate_diff = abs(linguistic['speech_rate'] - normal_speech_rate) / normal_speech_rate
    speech_rate_score = 1 - min(rate_diff, 1)
    aphasia_score += 0.4 * speech_rate_score
    
    # 2. 停顿频率评估 (权重: 0.3)
    normal_pause_freq = 0.5  # 正常停顿频率约0.5次/秒
    pause_diff = abs(linguistic['pause_frequency'] - normal_pause_freq) / normal_pause_freq
    pause_score = 1 - min(pause_diff, 1)
    aphasia_score += 0.3 * pause_score
    
    # 3. 流畅度评估 (权重: 0.3)
    fluency_score = 1 - min(advanced['jitter'] * 100, 1)  # 使用jitter作为流畅度指标
    aphasia_score += 0.3 * fluency_score
    
    # 风险等级判定
    def get_risk_level(score):
        if score >= 0.8: return "低"
        elif score >= 0.6: return "中低"
        elif score >= 0.4: return "中等"
        elif score >= 0.2: return "中高"
        else: return "高"
    
    # 计算综合评分
    # 1. 构音障碍评分（基于声学特征和音素准确率）
    articulation_score = 0
    # 基频稳定性 (30%)
    f0_std_score = 1 - min(basic['f0_std'] / 100, 1)
    articulation_score += 0.3 * f0_std_score
    
    # 谐波比率 (20%)
    harmonic_score = min(advanced['harmonic_ratio'], 1)
    articulation_score += 0.2 * harmonic_score
    
    # 音高变化 (20%)
    jitter_score = 1 - min(advanced['jitter'] * 100, 1)
    articulation_score += 0.2 * jitter_score
    
    # 音素准确率 (30%)
    phonetic = features.get('phonetic', None)
    if phonetic:
        vowel_score = phonetic['vowel_accuracy']
        consonant_score = phonetic['consonant_accuracy']
        phonetic_score = (vowel_score + consonant_score) / 2
        articulation_score += 0.3 * phonetic_score
    else:
        # 如果没有音素分析，调整其他特征的权重
        articulation_score = articulation_score / 0.7  # 归一化
    
    # 2. 失语症评分（基于语言学特征和音素准确率）
    aphasia_score = 0
    # 语速评估 (40%)
    normal_speech_rate = 4.0
    rate_diff = abs(linguistic['speech_rate'] - normal_speech_rate) / normal_speech_rate
    speech_rate_score = 1 - min(rate_diff, 1)
    aphasia_score += 0.4 * speech_rate_score
    
    # 停顿频率评估 (40%)
    normal_pause_freq = 0.5
    pause_diff = abs(linguistic['pause_frequency'] - normal_pause_freq) / normal_pause_freq
    pause_score = 1 - min(pause_diff, 1)
    aphasia_score += 0.4 * pause_score
    
    # 音素准确率对失语症评分的影响 (20%)
    if phonetic:
        aphasia_score += 0.2 * phonetic_score
    else:
        # 如果没有音素分析，调整其他特征的权重
        aphasia_score = aphasia_score / 0.8  # 归一化
    
    report = {
        'summary': f"音频时长: {duration:.2f}秒",
        'acoustic_features': {
            '基频稳定性': f"{basic['f0_std']:.2f} Hz (标准差)",
            '谐波失真': f"{advanced['jitter']:.4f} (微扰系数)",
            '谐波比率': f"{advanced['harmonic_ratio']:.2f}",
            '能量分布': "MFCC系数已计算"
        },
        'linguistic_features': {
            '语速': f"{linguistic['speech_rate']:.2f} 音节/秒",
            '停顿频率': f"{linguistic['pause_frequency']:.2f} 次/秒"
        }
    }
    
    # 添加音素分析结果（如果有）
    if phonetic:
        report['phonetic_features'] = {
            '元音准确率': f"{phonetic['vowel_accuracy']:.2%}",
            '辅音准确率': f"{phonetic['consonant_accuracy']:.2%}",
            '识别文本': phonetic['text'],
            '拼音分析': ' '.join(phonetic['pinyin'])
        }
    
    # 添加医学指标和评分
    report.update({
        'medical_indicators': {
            '构音障碍风险': get_risk_level(articulation_score),
            '失语症风险': get_risk_level(aphasia_score)
        },
        'scores': {
            '构音障碍评分': f"{articulation_score:.2f}",
            '失语症评分': f"{aphasia_score:.2f}",
            '详细指标': {
                '基频稳定性得分': f"{f0_std_score:.2f}",
                '谐波比率得分': f"{harmonic_score:.2f}",
                '音高变化得分': f"{jitter_score:.2f}",
                '语速评估得分': f"{speech_rate_score:.2f}",
                '停顿评估得分': f"{pause_score:.2f}"
            }
        }
    })
    
    # 如果有音素分析，添加相关得分
    if phonetic:
        report['scores']['详细指标'].update({
            '元音准确率得分': f"{vowel_score:.2f}",
            '辅音准确率得分': f"{consonant_score:.2f}",
            '音素综合得分': f"{phonetic_score:.2f}"
        })
    
    return report

# 示例医学特征计算（需添加到audio_processor.py）
# def calculate_clinical_features(y, sr):
#     return {
#         'vowel_articulation': analyze_vowel_articulation(y, sr),
#         'consonant_precision': analyze_consonant_precision(y, sr),
#         'speech_rate': calculate_syllable_rate(y, sr)
#     }

# 我在当前文件夹下实现了一个脑卒中早筛的语音识别系统，用于识别一段语音中与脑卒中相关的可量化特征，
# 如语速、停顿频率、元音辅音准确率等等。请为我分析这个系统的代码，并将项目结构写成一个readme.md存放在主文件夹下