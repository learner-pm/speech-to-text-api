import os
import json
import wave
import numpy as np
import soundfile as sf
from vosk import Model, KaldiRecognizer
import tempfile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self, model_path="model"):
        """初始化语音转文本服务
        
        Args:
            model_path (str): Vosk 模型路径
        """
        self.model_path = model_path
        self._ensure_model_dir()
        self.model = Model(model_path)
        logger.info("Vosk 模型加载完成")

    def _ensure_model_dir(self):
        """确保模型目录存在"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            logger.info(f"创建模型目录: {self.model_path}")

    def _convert_to_wav(self, input_path, output_path):
        """将音频文件转换为 WAV 格式
        
        Args:
            input_path (str): 输入文件路径
            output_path (str): 输出 WAV 文件路径
        """
        logger.info(f"开始转换音频文件: {input_path} -> {output_path}")
        # 使用 soundfile 读取音频
        data, samplerate = sf.read(input_path)
        logger.info(f"音频采样率: {samplerate}Hz, 通道数: {data.shape[1] if len(data.shape) > 1 else 1}")
        
        # 如果是立体声，转换为单声道
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            logger.info("已转换为单声道")
        
        # 确保数据是 16 位整数
        data = np.int16(data * 32767)
        
        # 使用 wave 写入 WAV 文件
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16 位
            wf.setframerate(samplerate)
            wf.writeframes(data.tobytes())
        logger.info("WAV 文件转换完成")

    def transcribe(self, audio_file):
        """将音频文件转换为文本
        
        Args:
            audio_file: 上传的音频文件对象
            
        Returns:
            dict: 包含识别结果的字典，格式为：
                {
                    'text': str,  # 识别的完整文本
                    'words': list  # 单词时间戳列表
                }
        """
        start_time = datetime.now()
        logger.info("开始语音识别处理")
        
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        logger.info(f"处理文件: {audio_file.filename} (类型: {file_extension})")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_input:
            temp_input_path = temp_input.name
            audio_file.save(temp_input_path)
            logger.info(f"临时文件已保存: {temp_input_path}")
        
        # 创建临时 WAV 文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            logger.info(f"创建临时 WAV 文件: {temp_wav_path}")
        
        try:
            # 转换为 WAV 格式
            self._convert_to_wav(temp_input_path, temp_wav_path)
            
            # 读取音频文件
            wf = None
            try:
                wf = wave.open(temp_wav_path, "rb")
                logger.info(f"音频信息 - 采样率: {wf.getframerate()}Hz, 通道数: {wf.getnchannels()}, 采样位数: {wf.getsampwidth()}")
                
                # 创建识别器
                rec = KaldiRecognizer(self.model, wf.getframerate())
                rec.SetWords(True)
                
                # 读取音频数据并识别
                logger.info("开始语音识别...")
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        pass
                        
                # 获取最终结果
                result = json.loads(rec.FinalResult())
                logger.info("语音识别完成")
                
                # 打印识别结果
                logger.info("=== 识别结果 ===")
                logger.info(f"完整文本: {result.get('text', '')}")
                if result.get('result'):
                    logger.info("单词时间戳:")
                    for word in result.get('result', []):
                        logger.info(f"  - {word['word']} ({word['start']:.2f}s - {word['end']:.2f}s)")
                logger.info("===============")
                
                # 计算处理时间
                process_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"总处理时间: {process_time:.2f}秒")
                
                return {
                    'text': result.get('text', ''),
                    'words': result.get('result', [])
                }
            finally:
                if wf is not None:
                    wf.close()
            
        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
            raise
        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                    logger.info(f"已删除临时输入文件: {temp_input_path}")
            except Exception as e:
                logger.error(f"删除临时输入文件时出错: {str(e)}")
                
            try:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                    logger.info(f"已删除临时 WAV 文件: {temp_wav_path}")
            except Exception as e:
                logger.error(f"删除临时 WAV 文件时出错: {str(e)}") 