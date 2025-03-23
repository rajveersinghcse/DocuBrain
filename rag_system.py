import os
import base64
import cv2
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from moviepy import VideoFileClip
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
import PyPDF2
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag_system")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"
PERSIST_DIR = "./storage"
TEMP_DIR = "./temp"
FRAME_SAMPLE_RATE = 3


class RagSystem:
    def __init__(self, model: str = MODEL, persist_dir: str = PERSIST_DIR, temp_dir: str = TEMP_DIR):
        self.model = model
        self.persist_dir = persist_dir
        self.temp_dir = temp_dir
        self.client = client

        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"RAG System initialized with model: {model}")

    def process_file(self, file_path: str, query: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
            return self._process_image(file_path, query)
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            return self._process_video(file_path, query)
        elif file_extension in ['.mp3', '.wav', '.ogg', '.flac']:
            return self._process_audio(file_path, query)
        elif file_extension == '.pdf':
            return self._process_pdf(file_path, query)
        elif file_extension in ['.csv', '.xlsx', '.xls']:
            return self._process_tabular(file_path, query)
        elif file_extension in ['.txt', '.md', '.json']:
            return self._process_text(file_path, query)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _process_image(self, image_path: str, query: str) -> Dict[str, Any]:
        logger.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        _, buffer = cv2.imencode(".jpg", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        summary = self._extract_image_summary(base64_image)
        answer = self._answer_query_with_summary(summary, query)
        
        return {
            "file_type": "image",
            "summary": summary,
            "answer": answer,
            "query": query
        }
    
    def _extract_image_summary(self, base64_image: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that generates detailed summaries of image content. "
                        "Provide a comprehensive summary, including key objects, main themes, and any notable details."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def _process_video(self, video_path: str, query: str) -> Dict[str, Any]:
        logger.info(f"Processing video: {video_path}")
        base64_frames, audio_path = self._extract_video_frames_and_audio(video_path)
        sampled_frames = base64_frames[::10]
        visual_summary = self._extract_video_summary(sampled_frames)
        transcript = self._transcribe_audio(audio_path)
        answer = self._answer_query_with_video_data(visual_summary, transcript, query)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {
            "file_type": "video",
            "visual_summary": visual_summary,
            "transcript": transcript.text if hasattr(transcript, 'text') else str(transcript),
            "answer": answer,
            "query": query
        }
    
    def _extract_video_frames_and_audio(self, video_path: str, seconds_per_frame: int = FRAME_SAMPLE_RATE) -> Tuple[List[str], str]:
        base64frames = []
        base_video_path, _ = os.path.splitext(video_path)
        audio_path = os.path.join(self.temp_dir, f"{os.path.basename(base_video_path)}.mp3")
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame = 0
        
        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            
            if not success:
                break
                
            _, buffer = cv2.imencode(".jpg", frame)
            base64frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip
            
        video.release()
        
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, bitrate="32k")
        clip.audio.close()
        clip.close()
        
        return base64frames, audio_path
    
    def _extract_video_summary(self, base64_frames: List[str]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that generates detailed summaries of video content. "
                        "Provide a comprehensive summary, including key events, main themes, and any notable details."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        "Here are the frames from the video:",
                        *map(
                            lambda x: {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpg;base64,{x}",
                                    "detail": "low",
                                },
                            },
                            base64_frames,
                        ),
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def _process_audio(self, audio_path: str, query: str) -> Dict[str, Any]:
        logger.info(f"Processing audio: {audio_path}")
        transcript = self._transcribe_audio(audio_path)
        answer = self._answer_query_with_audio_data(transcript, query)
        
        return {
            "file_type": "audio",
            "transcript": transcript.text if hasattr(transcript, 'text') else str(transcript),
            "answer": answer,
            "query": query
        }
    
    def _transcribe_audio(self, audio_path: str) -> Any:
        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript
    
    def _process_pdf(self, pdf_path: str, query: str) -> Dict[str, Any]:
        logger.info(f"Processing PDF: {pdf_path}")
        pdf_dir = os.path.join(self.temp_dir, os.path.basename(pdf_path).replace('.', '_'))
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_copy_path = os.path.join(pdf_dir, os.path.basename(pdf_path))
        shutil.copy2(pdf_path, pdf_copy_path)
        text_content = self._extract_text_from_pdf(pdf_path)
        
        try:
            documents = SimpleDirectoryReader(pdf_dir).load_data()
            index_dir = os.path.join(self.persist_dir, os.path.basename(pdf_path).replace('.', '_'))
            if os.path.exists(index_dir):
                storage_context = StorageContext.from_defaults(persist_dir=index_dir)
                index = VectorStoreIndex.from_storage(storage_context)
            else:
                index = VectorStoreIndex.from_documents(documents)
                os.makedirs(index_dir, exist_ok=True)
                index.storage_context.persist(persist_dir=index_dir)
            
            retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
            postprocessor = SimilarityPostprocessor(similarity_cutoff=0.70)
            
            query_engine = RetrieverQueryEngine(
                retriever=retriever, 
                node_postprocessors=[postprocessor]
            )
            
            response = query_engine.query(query)
            answer = str(response)
            
            shutil.rmtree(pdf_dir, ignore_errors=True)
            
            return {
                "file_type": "pdf",
                "text_sample": text_content[:500] + "..." if len(text_content) > 500 else text_content,
                "answer": answer,
                "query": query
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            shutil.rmtree(pdf_dir, ignore_errors=True)
            
            return self._process_text(pdf_path, query)
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text
    
    def _process_tabular(self, file_path: str, query: str) -> Dict[str, Any]:
        logger.info(f"Processing tabular file: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported tabular file type: {file_extension}")

        df_info = {
            "column_names": list(df.columns),
            "row_count": len(df),
            "column_count": len(df.columns),
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "first_rows": df.head(5).to_dict(orient='records'),
            "statistics": df.describe().to_dict() if df.select_dtypes(include=[np.number]).columns.any() else None
        }
        context = json.dumps(df_info, indent=2)
        answer = self._answer_query_with_tabular_data(context, query)
        
        return {
            "file_type": "tabular",
            "file_extension": file_extension,
            "columns": list(df.columns),
            "row_count": len(df),
            "answer": answer,
            "query": query
        }
    
    def _process_text(self, file_path: str, query: str) -> Dict[str, Any]:
        logger.info(f"Processing text file: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text_content = self._extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
        answer = self._answer_query_with_text(text_content, query)
        
        return {
            "file_type": "text",
            "file_extension": file_extension,
            "text_sample": text_content[:500] + "..." if len(text_content) > 500 else text_content,
            "answer": answer,
            "query": query
        }
    
    def _answer_query_with_summary(self, summary: str, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that answers questions based on image summaries. "
                        "Provide detailed and accurate answers based on the provided summary."
                    ),
                },
                {"role": "user", "content": f"Image Summary: {summary}\nQuery: {query}"},
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def _answer_query_with_video_data(self, visual_summary: str, transcript: Any, query: str) -> str:
        transcript_text = transcript.text if hasattr(transcript, 'text') else str(transcript)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that answers questions based on video content. "
                        "Provide detailed and accurate answers based on the provided visual summary and transcript."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Visual Summary: {visual_summary}\nTranscript: {transcript_text}\nQuery: {query}",
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def _answer_query_with_audio_data(self, transcript: Any, query: str) -> str:
        transcript_text = transcript.text if hasattr(transcript, 'text') else str(transcript)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that answers questions based on audio transcripts. "
                        "Provide detailed and accurate answers based on the provided transcript."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Audio Transcript: {transcript_text}\nQuery: {query}",
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def _answer_query_with_tabular_data(self, context: str, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that answers questions based on tabular data. "
                        "Provide detailed and accurate answers based on the provided data context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Data Context: {context}\nQuery: {query}",
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def _answer_query_with_text(self, text_content: str, query: str) -> str:
        max_tokens = 8000
        if len(text_content) > max_tokens * 4:
            text_content = text_content[:max_tokens * 4] + "...[truncated]"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that answers questions based on text content. "
                        "Provide detailed and accurate answers based on the provided text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Text Content: {text_content}\nQuery: {query}",
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)


if __name__ == "__main__":
    rag = RagSystem()
    rag.cleanup()