# AI-Powered Logistics Assistant 🚢

A comprehensive Streamlit web application that provides AI-powered assistance for logistics and supply chain management, featuring demand forecasting and analytics dashboards.

## Features

### 🤖 AI Chat Assistant
- Interactive chatbot powered by **Google Gemini AI**
- Specialized in logistics, shipping, and supply chain queries
- Real-time responses to questions about:
  - Shipping routes and transportation
  - Container logistics and port operations
  - Supply chain optimization
  - Trade regulations and customs
  - Risk management in logistics

### 📈 Demand Forecasting
- Time-series forecasting using Facebook Prophet
- Predict shipment demand for 7-90 days ahead
- Route-specific forecasting capabilities
- Confidence intervals and trend analysis
- Interactive visualizations with historical data

### 📊 Analytics Dashboard
- Comprehensive data visualizations
- Key performance indicators (KPIs)
- Demand trends and seasonal patterns
- Risk assessment by route and container type
- Monthly demand analysis
- Real-time data table with recent shipments

## Tech Stack

- **Frontend**: Streamlit
- **AI/LLM**: Google Gemini AI
- **Forecasting**: Facebook Prophet
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Matplotlib
- **Deployment**: Streamlit Cloud

## Installation & Setup

### 1. Clone the Repository

git clone <your-repo-url>
cd ai-logistics-assistant


### 2. Install Dependencies


### 3. Set Up API Keys
Create a `.streamlit/secrets.toml` file in your project directory:




To get a Google Gemini API key:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Add it to your secrets file

### 4. Run the Application


## Usage Guide

### Chat Assistant
1. Navigate to the "Chat Assistant" tab
2. Type logistics-related questions in the chat input
3. Use suggested questions for quick examples
4. View conversation history in the chat interface
5. Clear chat history using the sidebar button

### Demand Forecasting
1. Go to the "Demand Forecasting" tab
2. Select forecast period (7-90 days)
3. Choose a specific route or "All Routes"
4. Click "Generate Forecast" to create predictions
5. View interactive charts with confidence intervals
6. Analyze forecast metrics and insights

### Analytics Dashboard
1. Access the "Analytics Dashboard" tab
2. Use date range selectors to filter data
3. Review key performance indicators at the top
4. Explore various visualizations:
   - Daily demand trends
   - Route performance
   - Container type distribution
   - Risk assessment heatmaps
   - Seasonal patterns
5. Use table filters for detailed shipment information
6. Download filtered data as CSV

## Data

The application uses generated sample data that simulates realistic logistics scenarios with:
- 2 years of historical shipping data
- Multiple trade routes (Asia-Europe, Trans-Pacific, etc.)
- Various container types and port locations
- Risk assessments and cost information
- Seasonal demand patterns

For production use, replace the sample data generation with your actual logistics data.

## Features in Detail

### AI Chat Integration
- Uses Google Gemini's latest models for accurate responses
- Context-aware conversations about logistics topics
- Robust error handling and fallback responses
- Chat history preservation during session
- Specialized prompts for logistics domain

### Prophet Forecasting
- Handles seasonality (daily, weekly, yearly)
- Automatic trend detection
- Confidence intervals for predictions
- Customizable forecasting periods
- Route-specific analysis capabilities

### Interactive Visualizations
- Responsive Plotly charts
- Multiple chart types (line, bar, pie, heatmap)
- Real-time data updates
- Export capabilities
- Professional styling and animations

## Customization

### Adding New Features
- Extend the `LogisticsAssistant` class
- Add new tabs in the `run()` method
- Create additional visualization functions
- Integrate with other APIs or data sources

### Modifying Forecasting
- Adjust Prophet parameters in `create_forecast_model()`
- Add custom seasonality patterns
- Include additional regressors
- Implement ensemble forecasting methods

### Styling and Branding
- Modify CSS in the `st.markdown()` sections
- Change color schemes in Plotly charts
- Update icons and branding elements
- Add custom themes








