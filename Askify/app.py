import torch
from peft import PeftModel
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import requests
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import re
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.data_processor import DataProcessor
from src.rag.csv_chunker import CSVChunker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_engineer import PromptEngineer
from src.generation.qwen_client import QwenClient
from src.utils.config import Config
from src.utils.sql_executor import SQLExecutor
from src.utils.sql_validator import SQLValidator
from src.utils.lama_explainer import TinyLlamaExplainer

st.set_page_config(
    page_title="Askify - AI Data Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4f4f4f;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-right: 20%;
    }
    .sql-code {
        background: #2d3748;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #667eea;
    }
    .viz-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #00b09b;
        margin: 0.5rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

class BeautifulAskifyApp:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'data_processed' not in st.session_state:
            st.session_state.data_processed = False
        if 'processing_step' not in st.session_state:
            st.session_state.processing_step = 0
        if 'show_insights' not in st.session_state:
            st.session_state.show_insights = False
        if 'rag_system_initialized' not in st.session_state:
            st.session_state.rag_system_initialized = False
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ''
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
    
    def get_rag_system(self):
        if st.session_state.rag_system is None:
            st.session_state.rag_system = AskifyRAGSystem()
        return st.session_state.rag_system
    
    def initialize_rag_system(self):
        try:
            rag_system = self.get_rag_system()
            st.session_state.rag_system_initialized = True
            st.success("✅ RAG System initialized successfully!")
            return rag_system
                    
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            st.info("💡 Please check if all model files are available")
            raise e
    
    def _build_complete_response(self, sql_query: str, natural_language: str, validation_result: dict, execution_result: dict):
        response_parts = []
        
        if natural_language and natural_language.strip():
            clean_explanation = re.sub(r'[\{\}\[\]]', '', natural_language)
            clean_explanation = ' '.join(clean_explanation.split()) 
            clean_explanation = ' '.join(clean_explanation.split())
            
            if clean_explanation and clean_explanation[0].islower():
                clean_explanation = clean_explanation[0].upper() + clean_explanation[1:]
            
            if clean_explanation and not clean_explanation.endswith(('.', '!', '?')):
                clean_explanation += '.'

            response_parts.append(f"""
    <div style="background: linear-gradient(135deg, #00b09b, #96c93d); padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem;">
        <h3 style="margin: 0 0 0.5rem 0; color: white;">🤖 AI Insight</h3>
        <p style="margin: 0; color: white; font-size: 1.1rem; line-height: 1.6;">
            {clean_explanation}
        </p>
    </div>
    """)
        
        if execution_result and execution_result.get('success'):
            row_count = execution_result.get('row_count', 0)
            response_parts.append(f"""
    <div class='success-message'>
        ✅ <strong>Query Executed Successfully</strong><br>
        📊 Found {row_count} result{'s' if row_count != 1 else ''}
    </div>
    """)
        elif execution_result and not execution_result.get('success'):
            error_msg = execution_result.get('error', 'Query execution failed')
            suggestions = execution_result.get('suggestions', [])
            response_parts.append(f"""
    <div class='error-message'>
        ❌ <strong>Execution Error:</strong> {error_msg}<br>
        {'<br>💡 ' + '<br>💡 '.join(suggestions) if suggestions else ''}
    </div>
    """)
        
        response_parts.append(f"""
    <details style="margin: 1rem 0;">
        <summary style="cursor: pointer; font-weight: bold; padding: 0.5rem; background: #f8f9fa; border-radius: 5px;">
            📋 View SQL Query
        </summary>
        <div style="margin-top: 0.5rem;">
            <pre style="background: #2d3748; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto;">
    {sql_query}
            </pre>
        </div>
    </details>
    """)
        
        if validation_result.get('warnings'):
            warnings_html = '<br>'.join([f"⚠️ {w}" for w in validation_result['warnings']])
            response_parts.append(f"""
    <details style="margin: 0.5rem 0;">
        <summary style="cursor: pointer; color: #856404; padding: 0.5rem;">
            ⚠️ Validation Warnings
        </summary>
        <div class='warning-message' style="margin-top: 0.5rem;">
            {warnings_html}
        </div>
    </details>
    """)
        
        return "\n".join(response_parts)
    
    def show_landing_page(self):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="main-header">🤖 Askify</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Chat with Your Data • Get Instant Insights • Make Smarter Decisions</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center;">
                <h3 style="color: #667eea;">✨ Features</h3>
            </div>
            """, unsafe_allow_html=True)
            
            features = [
                {"icon": "🔍", "title": "Smart Search", "desc": "Natural language queries with RAG-powered context"},
                {"icon": "🚀", "title": "Lightning Fast", "desc": "Get answers in seconds with optimized retrieval"},
                {"icon": "🛡️", "title": "Privacy First", "desc": "Everything runs locally on your machine"},
                {"icon": "📊", "title": "Visual Insights", "desc": "Beautiful charts and data visualizations"},
                {"icon": "✅", "title": "SQL Validation", "desc": "Automatic SQL validation and execution"},
                {"icon": "🤖", "title": "AI Explanations", "desc": "TinyLlama powered natural language insights"}
            ]
            
            for feature in features:
                with st.container():
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>{feature['icon']} {feature['title']}</h4>
                        <p>{feature['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            lottie_url = "https://assets1.lottiefiles.com/packages/lf20_gn0tojcq.json"
            lottie_json = load_lottie_url(lottie_url)
            
            if lottie_json:
                st_lottie(lottie_json, height=300, key="landing-animation")
            else:
                st.markdown("""
                <div style="text-align: center; font-size: 8rem; margin: 2rem 0;">
                    📊➡️🤖➡️💡
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; margin-top: 2rem;">
                <h4 style="color: #667eea;">Ready to explore your data?</h4>
                <p>Upload a CSV file and start asking questions in plain English!</p>
            </div>
            """, unsafe_allow_html=True)
    
    def show_data_insights(self, df: pd.DataFrame):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #00b09b, #96c93d); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0; text-align: center;">📊 Data Insights & Visualizations</h2>
            <p style="text-align: center; margin: 0.5rem 0 0 0;">Explore your data through interactive charts and statistics</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", "🔍 Distributions", "📊 Correlations", "📋 Data Quality"
        ])
        
        with tab1:
            self._show_overview_insights(df)
        
        with tab2:
            self._show_distribution_insights(df)
        
        with tab3:
            self._show_correlation_insights(df)
        
        with tab4:
            self._show_data_quality_insights(df)
    
    def _show_overview_insights(self, df: pd.DataFrame):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Basic Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("### 🏷️ Data Information")
            info_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Memory Usage'],
                'Value': [
                    df.shape[0],
                    df.shape[1],
                    len(df.select_dtypes(include=[np.number]).columns),
                    len(df.select_dtypes(include=['object']).columns),
                    f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                ]
            }
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 📊 Column Types Distribution")
        col_types = {
            'Numeric': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical': len(df.select_dtypes(include=['object']).columns),
            'Boolean': len(df.select_dtypes(include=['bool']).columns),
            'Date/Time': len(df.select_dtypes(include=['datetime']).columns)
        }
        
        fig = px.pie(
            values=list(col_types.values()),
            names=list(col_types.keys()),
            title="Distribution of Column Data Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_distribution_insights(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            st.markdown("### 📈 Numeric Distributions")
            selected_numeric = st.selectbox("Select numeric column:", numeric_cols, key="numeric_dist")
            
            if selected_numeric:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(
                        df, x=selected_numeric,
                        title=f"Distribution of {selected_numeric}",
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(
                        df, y=selected_numeric,
                        title=f"Box Plot of {selected_numeric}",
                        color_discrete_sequence=['#764ba2']
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
        
        if len(categorical_cols) > 0:
            st.markdown("### 📊 Categorical Distributions")
            selected_categorical = st.selectbox("Select categorical column:", categorical_cols, key="cat_dist")
            
            if selected_categorical:
                value_counts = df[selected_categorical].value_counts().head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_bar = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top 10 Values in {selected_categorical}",
                        labels={'x': selected_categorical, 'y': 'Count'},
                        color_discrete_sequence=['#00b09b']
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    if len(value_counts) > 0:
                        fig_pie = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f"Top {len(value_counts)} Categories in {selected_categorical}",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
    
    def _show_correlation_insights(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            st.markdown("### 🔗 Correlation Analysis")
            
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📊 Scatter Plot Matrix")
            if len(numeric_cols) <= 6:
                fig_scatter = px.scatter_matrix(
                    df[numeric_cols].iloc[:, :6],
                    title="Scatter Plot Matrix",
                    height=800
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info(f"Showing scatter matrix for first 6 of {len(numeric_cols)} numeric columns")
                fig_scatter = px.scatter_matrix(
                    df[numeric_cols].iloc[:, :6],
                    title="Scatter Plot Matrix (First 6 Columns)",
                    height=800
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    def _show_data_quality_insights(self, df: pd.DataFrame):
        st.markdown("### 🧹 Data Quality Overview")
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        quality_data = {
            'Column': df.columns,
            'Missing Values': missing_data.values,
            'Missing %': missing_percent.values,
            'Data Type': df.dtypes.values
        }
        quality_df = pd.DataFrame(quality_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Missing Values by Column")
            st.dataframe(quality_df[['Column', 'Missing Values', 'Missing %']], use_container_width=True)
        
        with col2:
            if missing_data.sum() > 0:
                fig_missing = px.bar(
                    x=quality_df['Column'],
                    y=quality_df['Missing %'],
                    title="Missing Values Percentage by Column",
                    labels={'x': 'Columns', 'y': 'Missing %'},
                    color=quality_df['Missing %'],
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("🎉 No missing values found in your dataset!")
        
        st.markdown("#### 🔍 Duplicate Analysis")
        total_duplicates = df.duplicated().sum()
        duplicate_percent = (total_duplicates / len(df)) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Duplicate Rows", total_duplicates)
        with col3:
            st.metric("Duplicate %", f"{duplicate_percent:.2f}%")
    
    def show_chat_interface(self, df: pd.DataFrame):
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📊 Show Data Insights", use_container_width=True):
                st.session_state.show_insights = not st.session_state.show_insights
                st.rerun()
        
        if st.session_state.show_insights:
            self.show_data_insights(df)
            return
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0; text-align: center;">💬 Ask Anything About Your Data</h2>
            <p style="text-align: center; margin: 0.5rem 0 0 0;">I understand natural language - try questions like 'show top products by sales' or 'find customers with highest spending'</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Rows", df.shape[0])
        with col2:
            st.metric("🏷️ Total Columns", df.shape[1])
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("🔢 Numeric Columns", numeric_cols)
        with col4:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.metric("📝 Text Columns", categorical_cols)
        
        chat_container = st.container()
        
        st.markdown("### 💡 Try These Questions:")
        suggestions = [
            "Show me summary statistics",
            "What are the top 5 categories?",
            "Find correlations between numeric columns",
            "Show sales trends over time",
            "Which products have the highest revenue?"
        ]
        
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.current_question = suggestion
                    st.rerun()
        
        question = st.text_input(
            "💭 Your Question:",
            value=st.session_state.get('current_question', ''),
            placeholder="Type your question here...",
            key="question_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        
        if ask_button and question:
            with st.spinner("🔍 Searching through data and generating insights..."):
                self.process_question(question, df, chat_container)
    
    def process_question(self, question: str, df: pd.DataFrame, chat_container):
        st.session_state.messages.append({"role": "user", "content": question})
        
        try:
            rag_system = self.get_rag_system()
            
            if rag_system is None:
                raise RuntimeError("RAG system is not initialized. Please re-upload your file.")
            
            if not hasattr(rag_system, 'model_loaded') or not rag_system.model_loaded:
                raise RuntimeError("AI model is not loaded yet. Please wait for initialization to complete.")
            
            sql_query, context, retrieved_chunks, validation_result, execution_result = rag_system.ask_question(question, df)
            
            if execution_result and execution_result.get('success'):
                natural_language = rag_system.explainer.explain_results(
                    sql_query, 
                    execution_result.get('data', pd.DataFrame()), 
                    execution_result,
                    question
                )
            else:
                natural_language = rag_system.explainer.explain_results(
                    sql_query, 
                    pd.DataFrame(), 
                    execution_result or {},
                    question
                )
            
            print("=== FLAN-T5 EXPLANATION ===")
            print(f"Raw explanation: {natural_language}")
            print("===========================")
            
            full_response = self._build_complete_response(sql_query, natural_language, validation_result, execution_result)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sql_query": sql_query,
                "context": context,
                "retrieved_chunks": retrieved_chunks,
                "execution_result": execution_result,
                "validation_result": validation_result,
                "natural_language": natural_language
            })
            
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}",
                "error": True
            })
        
        self.display_messages(chat_container, df)
    
    def display_messages(self, container, df: pd.DataFrame):
        with container:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if message.get("error"):
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>Askify:</strong><br>
                            ❌ {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 Askify:</strong><br><br>
                            {message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        execution_result = message.get('execution_result')
                        if execution_result and execution_result.get('success') and execution_result.get('data') is not None:
                            result_df = execution_result['data']
                            if not result_df.empty:
                                st.markdown("#### 📊 Query Results:")
                                st.dataframe(result_df, use_container_width=True)
                                
                                if len(result_df) > 0 and len(result_df.select_dtypes(include=[np.number]).columns) > 0:
                                    self.create_visualizations(result_df)
                        
                        with st.expander("🔍 See how I found this answer"):
                            if 'context' in message:
                                st.markdown("**Relevant Data Context:**")
                                st.text_area("Context", message['context'], height=150, key=f"context_{i}", label_visibility="collapsed")
                            
                            if 'retrieved_chunks' in message:
                                st.markdown("**Top Retrieved Information:**")
                                for j, chunk in enumerate(message['retrieved_chunks'][:3]):
                                    st.write(f"**Chunk {j+1}** (Similarity: {chunk.get('similarity', 0):.3f})")
                                    st.text(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
                            
                            validation_result = message.get('validation_result')
                            if validation_result:
                                st.markdown("**SQL Validation Details:**")
                                if validation_result.get('is_valid'):
                                    st.success("✅ SQL is valid and safe to execute")
                                else:
                                    st.error(f"❌ SQL validation failed: {validation_result.get('error', 'Unknown error')}")
    
    def create_visualizations(self, result_df: pd.DataFrame):
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        categorical_cols = result_df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            if len(result_df) <= 20:
                fig = px.bar(result_df, x=cat_col, y=num_col, 
                           title=f"{num_col} by {cat_col}",
                           color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)
        
        elif len(numeric_cols) >= 2:
            fig = px.scatter(result_df, x=numeric_cols[0], y=numeric_cols[1],
                           title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
                           color_discrete_sequence=['#764ba2'])
            st.plotly_chart(fig, use_container_width=True)
        
        elif len(numeric_cols) == 1:
            fig = px.histogram(result_df, x=numeric_cols[0],
                             title=f"Distribution of {numeric_cols[0]}",
                             color_discrete_sequence=['#00b09b'])
            st.plotly_chart(fig, use_container_width=True)

def main():
    app = BeautifulAskifyApp()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
            <h2>📁 Upload Your Data</h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your dataset to start asking questions"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        1. **Upload** your CSV file
        2. **Wait** for AI processing
        3. **Ask** questions naturally
        4. **Get** instant SQL + insights
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        st.markdown("""
        - "Show top 10 customers"
        - "What's the average sales by category?"
        - "Find products with highest profit"
        - "Show monthly trends"
        - "Correlation between age and spending"
        """)
        
        if st.session_state.get('data_processed'):
            st.markdown("---")
            st.markdown("### 🔧 System Status")
            if st.session_state.get('rag_system_initialized'):
                st.success("✅ RAG System: Ready")
            else:
                st.error("❌ RAG System: Not Ready")
            
            rag_system = app.get_rag_system()
            if hasattr(rag_system, 'model_loaded') and rag_system.model_loaded:
                st.success("✅ AI Model: Loaded")
            else:
                st.warning("⚠️ AI Model: Not Loaded")
            
            if hasattr(rag_system, 'sql_validator'):
                st.success("✅ SQL Validator: Ready")
            if hasattr(rag_system, 'sql_executor'):
                st.success("✅ SQL Executor: Ready")
            if hasattr(rag_system, 'flant5_explainer'):
                st.success("✅ Tinyllama Explainer: Ready")
    
    if uploaded_file is None:
        app.show_landing_page()
    else:
        try:
            if not st.session_state.data_processed:
                st.info("🔄 Starting data processing...")
                
                with st.spinner("🔄 Initializing AI system..."):
                    rag_system = app.initialize_rag_system()
                
                with st.spinner("📊 Loading and processing data..."):
                    df = rag_system.processor.load_csv(uploaded_file)
                    chunk_count = rag_system.process_csv(df)
                    st.session_state.df = df
                
                with st.spinner("🤖 Loading AI model..."):
                    rag_system.initialize_model()
                    st.session_state.model_loaded = rag_system.model_loaded
                
                st.session_state.data_processed = True
                
                st.success("🎉 Data processed successfully! Ready for questions.")
                st.balloons()
            
            app.show_chat_interface(st.session_state.df)
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Make sure your CSV file is properly formatted and try again.")

class AskifyRAGSystem:
    def __init__(self):
        self.config = Config()
        self.processor = DataProcessor()
        self.chunker = CSVChunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.context_builder = ContextBuilder()
        self.prompt_engineer = PromptEngineer()
        self.qwen_client = None
        self.model_loaded = False
        
        self.sql_executor = SQLExecutor()
        self.sql_validator = SQLValidator()
        local_model_path = r"C:\Users\Mango\Desktop\nlpp\nlpp\models\tinylama"
        self.explainer = TinyLlamaExplainer(model_path=local_model_path)
        
    def initialize_model(self):
        try:
            if self.qwen_client is None:
                self.qwen_client = QwenClient()
                self.model_loaded = self.qwen_client.model_loaded
                if self.model_loaded:
                    st.success("✅ AI Model loaded successfully!")
                else:
                    st.error("❌ AI Model failed to load")
        except Exception as e:
            st.error(f"❌ Failed to load AI model: {e}")
            self.model_loaded = False
            raise e
    
    def process_csv(self, df: pd.DataFrame):
        try:
            chunks = self.chunker.chunk_dataframe(df)
            self.vector_store.add_chunks(chunks, self.embedder)
            st.success(f"✅ Processed {len(chunks)} data chunks")
            return len(chunks)
        except Exception as e:
            st.error(f"❌ Failed to process CSV: {e}")
            raise e
    
    def ask_question(self, question: str, df: pd.DataFrame = None):
        """Ask a question and get enhanced response with SQL validation and execution"""
        if not self.model_loaded:
            raise RuntimeError("AI model is not loaded yet. Please wait for initialization.")
        
        print(f"🤖 Processing question: {question[:50]}...")
        
        try:
            search_terms = self.context_builder.get_enhanced_search_terms(question)
            print(f"🔍 Search terms generated: {search_terms[:3]}")
            
            all_results = []
            for i, term in enumerate(search_terms[:3]):
                try:
                    print(f"   Searching with term {i+1}: '{term}'")
                    results = self.vector_store.search(term, self.embedder, top_k=3)
                    
                    for result in results:
                        result['search_term'] = term
                        result['search_rank'] = i + 1
                    
                    all_results.extend(results)
                    print(f"   → Found {len(results)} chunks")
                    
                except Exception as e:
                    print(f"   ⚠️  Search error for term '{term}': {e}")
            
            if not all_results:
                print("⚠️  No results from expanded terms, falling back to original question")
                all_results = self.vector_store.search(question, self.embedder, top_k=5)
                for result in all_results:
                    result['search_term'] = 'original'
                    result['search_rank'] = 1
            
            retrieved_chunks = self._deduplicate_and_sort_chunks(all_results)
            print(f"✅ Retrieved {len(retrieved_chunks)} unique chunks")
            
            if retrieved_chunks:
                top_chunk = retrieved_chunks[0]['content'][:150]
                print(f"📄 Top chunk preview: {top_chunk}...")
            
        except Exception as e:
            print(f"❌ Enhanced retrieval failed: {e}")
            print("🔄 Falling back to simple search...")
            retrieved_chunks = self.vector_store.search(question, self.embedder, top_k=5)
        
        context = self.context_builder.build_context(retrieved_chunks, question)
        
        print("📋 Context preview (first 300 chars):")
        print(context[:300] + "...\n")
        
        print("💭 Generating SQL query...")
        prompt = self.prompt_engineer.create_prompt(context, question)
        if hasattr(self.prompt_engineer, 'needs_joins_adapter'):
            if self.prompt_engineer.needs_joins_adapter(question):
                # Switch to JOIN adapter
                self.qwen_client.reload_with_joins()
                print("🔄 Using JOIN adapter for question")
            else:
                # Use main adapter
                self.qwen_client.reload_with_main()
                print("🔄 Using main SQL adapter")
        sql_query = self.qwen_client.generate_response(prompt)
        print(f"📝 Generated SQL: {sql_query[:100]}...")
        
        df_columns = list(df.columns) if df is not None else []
        validation_result = self.sql_validator.validate_sql(sql_query, df_columns)
        
        print(f"✅ SQL Validation: {validation_result.get('is_valid', False)}")
        if validation_result.get('warnings'):
            print(f"⚠️  Warnings: {validation_result['warnings'][:2]}")
        
        execution_result = None
        if validation_result.get('is_valid', False) and df is not None:
            print("⚡ Executing SQL...")
            try:
                execution_result = self.sql_executor.execute_sql(sql_query, df)
                if execution_result.get('success'):
                    print(f"✅ Execution successful: {execution_result.get('row_count', 0)} rows")
                else:
                    print(f"❌ Execution failed: {execution_result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Execution error: {e}")
                execution_result = {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        else:
            print("⏭️  Skipping execution (SQL not valid or no dataframe)")
        
        print("=" * 50)
        print("✅ Question processing complete\n")
        
        return sql_query, context, retrieved_chunks, validation_result, execution_result
    
    def _deduplicate_and_sort_chunks(self, chunks):
        if not chunks:
            return []
        
        content_groups = {}
        
        for chunk in chunks:
            content_preview = chunk.get('content', '')[:200]
            normalized = ' '.join(content_preview.lower().split())
            
            metadata = chunk.get('metadata', {})
            metadata_key = f"{metadata.get('type', '')}_{metadata.get('column', '')}"
            
            signature = f"{hash(normalized)}_{metadata_key}"
            
            if signature not in content_groups:
                content_groups[signature] = {
                    'content': chunk.get('content', ''),
                    'metadata': metadata,
                    'similarities': [],
                    'search_terms': set(),
                    'search_ranks': []
                }
            
            similarity = chunk.get('similarity', 0)
            content_groups[signature]['similarities'].append(similarity)
            
            if 'search_term' in chunk:
                content_groups[signature]['search_terms'].add(chunk['search_term'])
            
            if 'search_rank' in chunk:
                content_groups[signature]['search_ranks'].append(chunk['search_rank'])
        
        unique_chunks = []
        
        for signature, data in content_groups.items():
            avg_similarity = sum(data['similarities']) / len(data['similarities'])
            
            term_diversity = min(len(data['search_terms']) / 3, 1.0)
            
            if data['search_ranks']:
                avg_search_rank = sum(data['search_ranks']) / len(data['search_ranks'])
                rank_score = max(0, 1.0 - (avg_search_rank - 1) * 0.3)
            else:
                rank_score = 0.5
            
            combined_score = (
                avg_similarity * 0.6 +
                term_diversity * 0.2 +
                rank_score * 0.2
            )
            
            column_relevance = 0
            metadata = data.get('metadata', {})
            if 'column' in metadata:
                column_relevance = 0.1
            
            final_score = combined_score + column_relevance
            
            unique_chunks.append({
                'content': data['content'],
                'metadata': data['metadata'],
                'similarity': avg_similarity,
                'combined_score': final_score,
                'search_terms': list(data['search_terms']),
                'sources_count': len(data['similarities']),
                'avg_search_rank': avg_search_rank if data.get('search_ranks') else None
            })
        
        unique_chunks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return unique_chunks[:10]

if __name__ == "__main__":
    main()