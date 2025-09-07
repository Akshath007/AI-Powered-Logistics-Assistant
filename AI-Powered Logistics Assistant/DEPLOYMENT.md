# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Prepare Your Repository
Make sure your GitHub repository contains:
- `app.py` (main application file)
- `requirements.txt` (dependencies)
- `README.md` (documentation)

### 2. Deploy on Streamlit Cloud

1. **Visit Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io/)

2. **Connect GitHub**: Sign in with your GitHub account

3. **Create New App**: 
   - Click "New app"
   - Select your repository
   - Choose branch (usually `main` or `master`)
   - Set main file path: `app.py`
   - Click "Deploy!"

### 3. Configure Secrets (Important!)

1. **Access App Settings**: Once deployed, click on your app, then "⚙️ Settings"

2. **Add Secrets**: In the "Secrets" section, add:
   ```toml
   PERPLEXITY_API_KEY = "your_actual_api_key_here"
   ```

3. **Save**: Click "Save" to apply changes

### 4. Get Your Perplexity API Key

1. Visit [https://www.perplexity.ai/](https://www.perplexity.ai/)
2. Create an account or sign in
3. Go to API settings/dashboard
4. Generate a new API key
5. Copy the key and add it to Streamlit Cloud secrets

### 5. Test Your Deployment

1. Your app will be available at: `https://[your-app-name].streamlit.app/`
2. Test all three tabs:
   - Chat Assistant (requires API key)
   - Demand Forecasting (works with generated data)
   - Analytics Dashboard (works with generated data)

## Troubleshooting Common Issues

### App Won't Start
- Check the logs in Streamlit Cloud dashboard
- Verify all dependencies are in `requirements.txt`
- Ensure Python version compatibility

### Prophet Installation Issues
If Prophet fails to install, try this alternative `requirements.txt`:
```
streamlit>=1.28.0
pandas>=1.5.3
numpy>=1.24.3
plotly>=5.15.0
matplotlib>=3.7.1
requests>=2.31.0
prophet>=1.1.4
pystan>=3.3.0
```

### Chat Not Working
- Verify your Perplexity API key is correctly set in Streamlit Cloud secrets
- Check API key quotas and billing status
- Ensure no extra spaces in the API key

### Forecasting Errors
- Prophet requires at least 2 data points
- Check for missing or invalid dates in data
- Verify Prophet installation in the deployment logs

## App Configuration

The app includes a `config.toml` file that should be placed in `.streamlit/` folder for local development:

```
.streamlit/
├── config.toml
└── secrets.toml
```

For Streamlit Cloud, configuration is handled automatically, but secrets must be added manually.

## Performance Tips

1. **Data Caching**: The app uses Streamlit's session state for data persistence
2. **API Limits**: Be aware of Perplexity API rate limits
3. **Resource Usage**: Prophet forecasting can be CPU intensive

## Custom Domain (Optional)

To use a custom domain:
1. Upgrade to Streamlit Cloud Pro
2. Follow the custom domain setup guide
3. Configure DNS settings

## Environment Variables

For local development, create `.streamlit/secrets.toml`:
```toml
PERPLEXITY_API_KEY = "your_key_here"
```

For production deployment, use Streamlit Cloud's secrets management.

## Monitoring and Analytics

- Use Streamlit Cloud's built-in analytics
- Monitor API usage on Perplexity dashboard
- Check app performance metrics

---

**Need Help?** 
- Streamlit Cloud Docs: [docs.streamlit.io](https://docs.streamlit.io/)
- Community Forum: [discuss.streamlit.io](https://discuss.streamlit.io/)
- Perplexity API Docs: [docs.perplexity.ai](https://docs.perplexity.ai/)