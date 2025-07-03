import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# File handling libraries
import PyPDF2
import docx
from PIL import Image
import pytesseract
import io
import base64
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import os
import tempfile

# Together.ai API
from together import Together

# Set page config
st.set_page_config(
    page_title="🤖 AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalystAgent:
    """
    Intelligent Data Analyst Agent that can:
    - Process multiple file types (.csv, .xlsx, .pdf, .docx, .txt, images)
    - Perform comprehensive data analysis
    - Create intelligent visualizations
    - Answer questions about the data
    - Provide insights and recommendations
    """
    
    def __init__(self, together_api_key: str):
        """Initialize the agent with Together.ai API key"""
        self.together_api_key = together_api_key
        self.client = Together(api_key=together_api_key)
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        
        # Storage for processed data
        self.data = None
        self.data_info = {}
        self.file_type = None
        self.analysis_context = ""
        self.conversation_history = []
        
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call the Llama model via Together.ai API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst AI assistant. Provide clear, accurate, and insightful analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """Load and process different file types"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            self.file_type = file_extension
            
            if file_extension == '.csv':
                return self._load_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return self._load_excel(file_path)
            elif file_extension == '.pdf':
                return self._load_pdf(file_path)
            elif file_extension == '.docx':
                return self._load_docx(file_path)
            elif file_extension == '.txt':
                return self._load_txt(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                return self._load_image(file_path)
            else:
                return {"error": f"Unsupported file type: {file_extension}"}
                
        except Exception as e:
            return {"error": f"Error loading file: {str(e)}"}
    
    def _load_csv(self, file_path: str) -> Dict[str, Any]:
        """Load CSV file"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            separators = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:
                            self.data = df
                            return self._analyze_dataframe(df, "CSV")
                    except:
                        continue
            
            df = pd.read_csv(file_path)
            self.data = df
            return self._analyze_dataframe(df, "CSV")
            
        except Exception as e:
            return {"error": f"Error loading CSV: {str(e)}"}
    
    def _load_excel(self, file_path: str) -> Dict[str, Any]:
        """Load Excel file"""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names
            
            if len(sheets) == 1:
                df = pd.read_excel(file_path)
                self.data = df
                return self._analyze_dataframe(df, "Excel")
            else:
                all_data = {}
                for sheet in sheets:
                    all_data[sheet] = pd.read_excel(file_path, sheet_name=sheet)
                
                self.data = all_data[sheets[0]]
                result = self._analyze_dataframe(all_data[sheets[0]], "Excel")
                result['additional_info'] = f"File contains {len(sheets)} sheets: {', '.join(sheets)}"
                return result
                
        except Exception as e:
            return {"error": f"Error loading Excel: {str(e)}"}
    
    def _load_pdf(self, file_path: str) -> Dict[str, Any]:
        """Load PDF file and extract text"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            self.data = text
            return self._analyze_text(text, "PDF")
            
        except Exception as e:
            return {"error": f"Error loading PDF: {str(e)}"}
    
    def _load_docx(self, file_path: str) -> Dict[str, Any]:
        """Load DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            self.data = text
            return self._analyze_text(text, "DOCX")
            
        except Exception as e:
            return {"error": f"Error loading DOCX: {str(e)}"}
    
    def _load_txt(self, file_path: str) -> Dict[str, Any]:
        """Load text file"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            self.data = text
            return self._analyze_text(text, "TXT")
            
        except Exception as e:
            return {"error": f"Error loading TXT: {str(e)}"}
    
    def _load_image(self, file_path: str) -> Dict[str, Any]:
        """Load image file and extract text using OCR"""
        try:
            image = Image.open(file_path)
            
            try:
                text = pytesseract.image_to_string(image)
            except:
                text = "OCR extraction failed. Image loaded successfully but text extraction not available."
            
            self.data = {"image_path": file_path, "extracted_text": text}
            
            return {
                "file_type": "Image",
                "image_size": image.size,
                "image_mode": image.mode,
                "extracted_text_length": len(text),
                "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
                "analysis": self._call_llm(f"Analyze this text extracted from an image:\n\n{text[:1000]}")
            }
            
        except Exception as e:
            return {"error": f"Error loading image: {str(e)}"}
    
    def _analyze_dataframe(self, df: pd.DataFrame, file_type: str) -> Dict[str, Any]:
        """Comprehensive analysis of DataFrame"""
        try:
            analysis = {
                "file_type": file_type,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "sample_data": df.head().to_dict(),
            }
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                analysis["categorical_info"] = {}
                for col in categorical_cols:
                    analysis["categorical_info"][col] = {
                        "unique_count": df[col].nunique(),
                        "top_values": df[col].value_counts().head().to_dict()
                    }
            
            context = f"""
            Dataset Analysis:
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Columns: {', '.join(df.columns.tolist())}
            - Data types: {dict(df.dtypes)}
            - Missing values: {dict(df.isnull().sum())}
            - Sample data: {df.head(3).to_string()}
            """
            
            analysis["ai_insights"] = self._call_llm(
                f"Provide comprehensive insights about this dataset:\n{context}\n\nWhat patterns, trends, or interesting findings can you identify? What analysis would be most valuable?",
                max_tokens=1500
            )
            
            self.analysis_context = context
            self.data_info = analysis
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing DataFrame: {str(e)}"}
    
    def _analyze_text(self, text: str, file_type: str) -> Dict[str, Any]:
        """Analyze text content"""
        try:
            analysis = {
                "file_type": file_type,
                "text_length": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.split('\n')),
                "character_count": len(text),
                "preview": text[:1000] + "..." if len(text) > 1000 else text
            }
            
            analysis["ai_insights"] = self._call_llm(
                f"Analyze this {file_type} document content:\n\n{text[:2000]}\n\nProvide key insights, themes, and summary.",
                max_tokens=1500
            )
            
            self.analysis_context = f"Document type: {file_type}\nContent length: {len(text)} characters\nPreview: {text[:500]}"
            self.data_info = analysis
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing text: {str(e)}"}
    
    def answer_question(self, question: str) -> str:
        """Answer questions about the loaded data"""
        if self.data is None:
            return "No data has been loaded yet. Please load a file first."
        
        try:
            self.conversation_history.append({"type": "question", "content": question})
            
            if isinstance(self.data, pd.DataFrame):
                context = self._prepare_dataframe_context(question)
            else:
                context = self._prepare_text_context(question)
            
            prompt = f"""
            You are analyzing data for a user. Here's the context:
            
            {context}
            
            User Question: {question}
            
            Previous conversation: {self.conversation_history[-3:] if len(self.conversation_history) > 1 else "None"}
            
            Provide a comprehensive, accurate answer. If the question requires specific calculations or analysis, perform them step by step.
            """
            
            answer = self._call_llm(prompt, max_tokens=1500)
            self.conversation_history.append({"type": "answer", "content": answer})
            
            return answer
            
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def _prepare_dataframe_context(self, question: str) -> str:
        """Prepare context for DataFrame questions"""
        df = self.data
        context = f"""
        Dataset Information:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Columns: {', '.join(df.columns.tolist())}
        - Data types: {dict(df.dtypes)}
        - Missing values: {dict(df.isnull().sum())}
        
        Sample data (first 5 rows):
        {df.head().to_string()}
        
        Statistical summary for numeric columns:
        {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else "No numeric columns"}
        """
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['correlation', 'relationship', 'related']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                context += f"\nCorrelation matrix:\n{df[numeric_cols].corr().to_string()}\n"
        
        return context
    
    def _prepare_text_context(self, question: str) -> str:
        """Prepare context for text-based questions"""
        if isinstance(self.data, dict) and 'extracted_text' in self.data:
            return f"""
            Data Type: Image with extracted text
            Text content: {self.data['extracted_text'][:2000]}
            """
        else:
            return f"""
            Data Type: Text document ({self.file_type})
            Content length: {len(self.data)} characters
            Content preview: {self.data[:2000]}
            """

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Data Analyst Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Powered by Llama-4-Maverick-17B-128E-Instruct-FP8 via Together.ai**")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for API key and file upload
    with st.sidebar:
        st.markdown("### 🔑 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Enter your Together.ai API Key:",
            type="password",
            help="Get your free API key from https://www.together.ai/"
        )
        
        if api_key:
            if st.session_state.agent is None:
                try:
                    st.session_state.agent = DataAnalystAgent(api_key)
                    st.success("✅ Agent initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing agent: {str(e)}")
        
        st.markdown("### 📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a file to analyze:",
            type=['csv', 'xlsx', 'xls', 'pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: CSV, Excel, PDF, Word, Text, and Images"
        )
        
        if uploaded_file is not None and st.session_state.agent is not None:
            if st.button("🔍 Analyze File", type="primary"):
                with st.spinner("Analyzing file..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Analyze the file
                    result = st.session_state.agent.load_file(tmp_file_path)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.session_state.analysis_result = result
                        st.session_state.analysis_complete = True
                        st.success("✅ File analyzed successfully!")
                        st.rerun()
    
    # Main content area
    if not api_key:
        st.info("👈 Please enter your Together.ai API key in the sidebar to get started.")
        st.markdown("""
        ### How to get started:
        1. **Get API Key**: Visit [Together.ai](https://www.together.ai/) and create a free account
        2. **Enter API Key**: Paste it in the sidebar
        3. **Upload File**: Choose from CSV, Excel, PDF, Word, Text, or Image files
        4. **Analyze**: Click the analyze button and explore your data!
        
        ### Supported File Types:
        - **Structured Data**: CSV, Excel (.xlsx, .xls)
        - **Documents**: PDF, Word (.docx), Text (.txt)
        - **Images**: PNG, JPG, JPEG, BMP, TIFF (with OCR text extraction)
        """)
        return
    
    if not st.session_state.analysis_complete:
        st.info("👈 Please upload a file in the sidebar to begin analysis.")
        
        # Show example of what the agent can do
        st.markdown("### 🎯 What this agent can do:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Data Analysis**
            - Comprehensive EDA
            - Statistical summaries
            - Data quality assessment
            - Missing value analysis
            """)
        
        with col2:
            st.markdown("""
            **📈 Visualizations**
            - Interactive charts
            - Correlation matrices
            - Distribution plots
            - Custom visualizations
            """)
        
        with col3:
            st.markdown("""
            **🤖 AI-Powered Insights**
            - Intelligent Q&A
            - Pattern recognition
            - Recommendations
            - Natural language analysis
            """)
        
        return
    
    # Display analysis results
    result = st.session_state.analysis_result
    
    # Analysis Overview
    st.markdown('<h2 class="section-header">📊 Analysis Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("File Type", result.get('file_type', 'Unknown'))
    
    if 'shape' in result:
        with col2:
            st.metric("Rows", f"{result['shape'][0]:,}")
        with col3:
            st.metric("Columns", result['shape'][1])
        with col4:
            missing_pct = sum(result.get('missing_values', {}).values()) / (result['shape'][0] * result['shape'][1]) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # AI Insights
    if 'ai_insights' in result:
        st.markdown('<h2 class="section-header">🤖 AI Insights</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">{result["ai_insights"]}</div>', unsafe_allow_html=True)
    
    # Data Preview (for structured data)
    if isinstance(st.session_state.agent.data, pd.DataFrame):
        st.markdown('<h2 class="section-header">👀 Data Preview</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "📈 Summary Statistics", "🔍 Data Info"])
        
        with tab1:
            st.dataframe(st.session_state.agent.data.head(10), use_container_width=True)
        
        with tab2:
            numeric_cols = st.session_state.agent.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(st.session_state.agent.data[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary.")
        
        with tab3:
            info_df = pd.DataFrame({
                'Column': st.session_state.agent.data.columns,
                'Data Type': st.session_state.agent.data.dtypes.astype(str),
                'Non-Null Count': st.session_state.agent.data.count(),
                'Null Count': st.session_state.agent.data.isnull().sum(),
                'Unique Values': [st.session_state.agent.data[col].nunique() for col in st.session_state.agent.data.columns]
            })
            st.dataframe(info_df, use_container_width=True)
    
    # Visualizations
    if isinstance(st.session_state.agent.data, pd.DataFrame):
        st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)
        
        df = st.session_state.agent.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            # Correlation heatmap
            if len(numeric_cols) > 1:
                st.subheader("🔥 Correlation Matrix")
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
            
            # Distribution plots
            st.subheader("📊 Distribution Analysis")
            selected_numeric = st.selectbox("Select numeric column:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=selected_numeric, title=f"Box Plot of {selected_numeric}")
                st.plotly_chart(fig, use_container_width=True)
        
        if len(categorical_cols) > 0:
            st.subheader("📋 Categorical Analysis")
            selected_categorical = st.selectbox("Select categorical column:", categorical_cols)
            
            if df[selected_categorical].nunique() <= 20:
                fig = px.bar(x=df[selected_categorical].value_counts().index, 
                           y=df[selected_categorical].value_counts().values,
                           title=f"Distribution of {selected_categorical}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Column '{selected_categorical}' has too many unique values ({df[selected_categorical].nunique()}) to display effectively.")
    
    # Q&A Section
    st.markdown('<h2 class="section-header">❓ Ask Questions About Your Data</h2>', unsafe_allow_html=True)
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        if chat['type'] == 'question':
            st.markdown(f"**🙋 You:** {chat['content']}")
        else:
            st.markdown(f"**🤖 Agent:** {chat['content']}")
        st.markdown("---")
    
    # Question input
    user_question = st.text_input("Ask a question about your data:", key="question_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🚀 Ask", type="primary")
    
    if ask_button and user_question:
        with st.spinner("Thinking..."):
            answer = st.session_state.agent.answer_question(user_question)
            
            # Add to chat history
            st.session_state.chat_history.append({"type": "question", "content": user_question})
            st.session_state.chat_history.append({"type": "answer", "content": answer})
            
            st.rerun()
    
    # Suggested questions
    if isinstance(st.session_state.agent.data, pd.DataFrame):
        st.markdown("### 💡 Suggested Questions:")
        
        suggested_questions = [
            "What are the main patterns in this data?",
            "Are there any outliers or anomalies?",
            "What correlations exist between variables?",
            "What insights can you provide about this dataset?",
            "What would be good next steps for analysis?"
        ]
        
        # Add specific questions based on data
        df = st.session_state.agent.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            suggested_questions.append(f"What is the average {numeric_cols[0]}?")
        
        if len(categorical_cols) > 0:
            suggested_questions.append(f"What is the distribution of {categorical_cols[0]}?")
        
        cols = st.columns(3)
        for i, question in enumerate(suggested_questions[:6]):
            with cols[i % 3]:
                if st.button(question, key=f"suggested_{i}"):
                    with st.spinner("Thinking..."):
                        answer = st.session_state.agent.answer_question(question)
                        
                        st.session_state.chat_history.append({"type": "question", "content": question})
                        st.session_state.chat_history.append({"type": "answer", "content": answer})
                        
                        st.rerun()

if __name__ == "__main__":
    main()