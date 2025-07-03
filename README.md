
# 🤖 AI Data Analyst Agent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Together.ai](https://img.shields.io/badge/Together.ai-API-green.svg)](https://www.together.ai/)


An intelligent data analysis agent powered by **Llama-4-Maverick-17B-128E-Instruct-FP8** that can process multiple file types, perform comprehensive data analysis, create visualizations, and answer questions about your data using natural language.

## 🌟 Features

### 📁 Multi-Format File Support
- **Structured Data**: CSV, Excel (.xlsx, .xls)
- **Documents**: PDF, Word (.docx), Text (.txt)
- **Images**: PNG, JPG, JPEG, BMP, TIFF (with OCR text extraction)

### 🔍 Comprehensive Data Analysis
- **Exploratory Data Analysis (EDA)**: Automated statistical summaries and data profiling
- **Data Quality Assessment**: Missing value analysis and data type detection
- **Pattern Recognition**: Correlation analysis and trend identification
- **AI-Powered Insights**: Intelligent observations and recommendations

### 📊 Interactive Visualizations
- **Statistical Plots**: Histograms, box plots, distribution analysis
- **Correlation Heatmaps**: Visual correlation matrices for numeric data
- **Categorical Analysis**: Bar charts and frequency distributions
- **Interactive Charts**: Built with Plotly for enhanced user experience

### 🤖 Natural Language Q&A
- **Conversational Interface**: Ask questions about your data in plain English
- **Context-Aware Responses**: Maintains conversation history for better understanding
- **Intelligent Suggestions**: Provides relevant follow-up questions
- **Multi-Turn Conversations**: Supports complex analytical discussions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Together.ai API key (free account available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-data-analyst-agent.git
   cd ai-data-analyst-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get your Together.ai API key**
   - Visit [Together.ai](https://www.together.ai/)
   - Create a free account
   - Generate your API key from the dashboard

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL manually

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.15.0
together>=0.2.0
PyPDF2>=3.0.0
python-docx>=0.8.11
Pillow>=9.0.0
pytesseract>=0.3.10
openpyxl>=3.1.0
xlrd>=2.0.0
```

## 🔧 Usage

### 1. Launch the Application
```bash
streamlit run app.py
```

### 2. Configure API Key
- Enter your Together.ai API key in the sidebar
- The agent will initialize automatically

### 3. Upload Your Data
- Use the file uploader in the sidebar
- Supported formats: CSV, Excel, PDF, Word, Text, Images
- Click "Analyze File" to process your data

### 4. Explore Your Data
- **Analysis Overview**: View key statistics and metrics
- **AI Insights**: Get intelligent observations about your data
- **Data Preview**: Browse sample data and summary statistics
- **Visualizations**: Explore interactive charts and graphs

### 5. Ask Questions
- Use the Q&A section to ask questions in natural language
- Try suggested questions or ask your own
- The agent maintains conversation context for follow-up questions

## 💡 Example Use Cases

### Business Analytics
```
"What are the top performing products this quarter?"
"Are there any seasonal trends in our sales data?"
"Which customer segments show the highest retention rates?"
```

### Scientific Research
```
"What correlations exist between these variables?"
"Are there any outliers in the experimental results?"
"What statistical significance do these findings have?"
```

### Financial Analysis
```
"What is the monthly growth rate trend?"
"Which factors most influence our revenue?"
"Are there any anomalies in the transaction data?"
```

## 🏗️ Architecture

### Core Components

1. **DataAnalystAgent Class**
   - Main orchestrator for all analysis operations
   - Handles file processing and LLM interactions
   - Maintains conversation context and data state

2. **File Processing Pipeline**
   - Multi-format file loaders with encoding detection
   - OCR integration for image text extraction
   - Error handling and fallback mechanisms

3. **Analysis Engine**
   - Statistical analysis and data profiling
   - Pattern recognition and anomaly detection
   - AI-powered insight generation

4. **Visualization System**
   - Interactive charts with Plotly
   - Statistical plots with Matplotlib/Seaborn
   - Responsive design for different screen sizes

5. **Conversational Interface**
   - Natural language processing for user queries
   - Context-aware response generation
   - Multi-turn conversation support

## 🔒 Security & Privacy

- **API Key Security**: API keys are handled securely and not stored permanently
- **Local Processing**: Files are processed locally and temporarily
- **No Data Persistence**: Uploaded data is not stored between sessions
- **Secure Communication**: All API communications use HTTPS

## 🛠️ Customization

### Adding New File Types
```python
def _load_new_format(self, file_path: str) -> Dict[str, Any]:
    """Load custom file format"""
    # Implement your custom loader here
    pass
```

### Custom Visualizations
```python
# Add custom visualization functions
def create_custom_chart(df, column):
    fig = px.custom_chart(df, x=column)
    return fig
```

### Model Configuration
```python
# Change the LLM model
self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your Together.ai API key is valid
   - Check your API usage limits
   - Verify internet connectivity

2. **File Upload Problems**
   - Check file size limits (Streamlit default: 200MB)
   - Ensure file format is supported
   - Try different encoding if CSV fails to load

3. **OCR Not Working**
   - Install Tesseract OCR system dependency
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Memory Issues**
   - Large files may require more RAM
   - Consider sampling large datasets
   - Use chunked processing for very large files

## 📈 Performance Tips

- **Large Files**: Sample your data for faster analysis
- **Memory Usage**: Close other applications when processing large datasets
- **API Limits**: Be mindful of Together.ai rate limits
- **Caching**: Use Streamlit's caching features for repeated operations

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/yourusername/ai-data-analyst-agent.git
cd ai-data-analyst-agent
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If you have dev dependencies
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Together.ai** for providing the LLM API
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **OpenAI** for inspiration in AI agent design
- **Meta** for the Llama model architecture


## 🔮 Roadmap

- [ ] **Advanced Analytics**: Time series analysis, forecasting
- [ ] **More File Types**: JSON, XML, Parquet support
- [ ] **Database Integration**: SQL database connections
- [ ] **Export Features**: Report generation and export
- [ ] **Collaboration**: Multi-user support and sharing
- [ ] **Advanced Visualizations**: 3D plots, interactive dashboards
- [ ] **API Endpoints**: REST API for programmatic access
- [ ] **Mobile Support**: Responsive design improvements

---

⭐ **Star this repository if you find it helpful!** ⭐

Built with ❤️ using Streamlit and Together.ai
