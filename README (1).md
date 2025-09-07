# AI-Powered Logistics Assistant 🚢

A comprehensive Streamlit web application that provides AI-powered assistance for logistics and supply chain management, featuring demand forecasting and analytics dashboards.

## Features

### 🤖 AI Chat Assistant
- Interactive chatbot powered by Perplexity AI API
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
- **AI/LLM**: Perplexity AI API
- **Forecasting**: Facebook Prophet
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Matplotlib
- **Deployment**: Streamlit Cloud

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ai-logistics-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys
Create a `.streamlit/secrets.toml` file in your project directory:
```toml
PERPLEXITY_API_KEY = "your_perplexity_api_key_here"
```

To get a Perplexity API key:
1. Sign up at [https://www.perplexity.ai/](https://www.perplexity.ai/)
2. Navigate to API settings
3. Generate a new API key
4. Add it to your secrets file

### 4. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Deployment on Streamlit Cloud

### 1. Push to GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Connect your GitHub account
3. Select your repository
4. Set the main file path: `app.py`
5. Add your secrets in the app settings:
   - Key: `PERPLEXITY_API_KEY`
   - Value: Your Perplexity API key

### 3. Advanced Configuration
The app includes a `.streamlit/config.toml` file for additional configuration options.

## Project Structure

```
ai-logistics-assistant/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore rules
└── .streamlit/
    └── config.toml      # Streamlit configuration
```

## Usage Guide

### Chat Assistant
1. Navigate to the "Chat Assistant" tab
2. Type logistics-related questions in the chat input
3. Use suggested questions for quick examples
4. View conversation history in the chat interface

### Demand Forecasting
1. Go to the "Demand Forecasting" tab
2. Select forecast period (7-90 days)
3. Choose a specific route or "All Routes"
4. Click "Generate Forecast" to create predictions
5. View interactive charts with confidence intervals
6. Analyze forecast metrics and insights

### Analytics Dashboard
1. Access the "Analytics Dashboard" tab
2. Review key performance indicators at the top
3. Explore various visualizations:
   - Daily demand trends
   - Route performance
   - Container type distribution
   - Risk assessment
   - Seasonal patterns
4. Use the data table for detailed shipment information

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
- Uses Perplexity AI's latest models for accurate responses
- Context-aware conversations about logistics topics
- Error handling and fallback responses
- Chat history preservation during session

### Prophet Forecasting
- Handles seasonality (daily, weekly, yearly)
- Automatic trend detection
- Confidence intervals for predictions
- Customizable forecasting periods

### Interactive Visualizations
- Responsive Plotly charts
- Multiple chart types (line, bar, pie, heatmap)
- Real-time data updates
- Export capabilities

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

## Troubleshooting

### Common Issues
1. **Prophet Installation**: If Prophet fails to install, try:
   ```bash
   conda install -c conda-forge prophet
   ```

2. **API Key Issues**: Ensure your Perplexity API key is correctly set in secrets

3. **Memory Issues**: For large datasets, consider data sampling or pagination

4. **Deployment Errors**: Check Streamlit Cloud logs and ensure all dependencies are in requirements.txt

### Performance Optimization
- Use `@st.cache_data` for expensive computations
- Implement data pagination for large datasets
- Optimize Prophet model parameters
- Use efficient data structures

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ using Streamlit, Prophet, and Perplexity AI**