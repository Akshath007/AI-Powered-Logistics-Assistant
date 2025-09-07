import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, handle if not installed
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    st.error("Prophet not installed. Please install it using: pip install prophet")
    PROPHET_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI-Powered Logistics Assistant",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tab-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

class LogisticsAssistant:
    """Main class for the AI-Powered Logistics Assistant"""
    
    def __init__(self):
        self.initialize_session_state()
        self.generate_sample_data()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'sample_data' not in st.session_state:
            st.session_state.sample_data = None
        if 'forecast_data' not in st.session_state:
            st.session_state.forecast_data = None
    
    def generate_sample_data(self):
        """Generate sample shipping/logistics data for demonstration"""
        if st.session_state.sample_data is None:
            np.random.seed(42)
            
            # Generate 2 years of daily data
            start_date = datetime.now() - timedelta(days=730)
            dates = pd.date_range(start=start_date, periods=730, freq='D')
            
            # Create realistic shipping demand with seasonal patterns
            base_demand = 1000
            seasonal_factor = 200 * np.sin(2 * np.pi * np.arange(730) / 365.25)
            weekly_factor = 50 * np.sin(2 * np.pi * np.arange(730) / 7)
            noise = np.random.normal(0, 50, 730)
            trend = np.linspace(0, 300, 730)  # Growing trend
            
            shipment_demand = base_demand + seasonal_factor + weekly_factor + trend + noise
            shipment_demand = np.maximum(shipment_demand, 100)  # Ensure positive values
            
            # Create additional features
            routes = ['Asia-Europe', 'Trans-Pacific', 'Atlantic', 'Intra-Asia', 'Mediterranean']
            container_types = ['20ft Standard', '40ft Standard', '40ft High Cube', 'Refrigerated']
            ports = ['Shanghai', 'Singapore', 'Rotterdam', 'Los Angeles', 'Hamburg']
            
            data = []
            for i, date in enumerate(dates):
                for _ in range(np.random.randint(5, 15)):  # 5-15 shipments per day
                    data.append({
                        'date': date,
                        'shipment_demand': max(50, int(shipment_demand[i] + np.random.normal(0, 100))),
                        'route': np.random.choice(routes),
                        'container_type': np.random.choice(container_types),
                        'origin_port': np.random.choice(ports),
                        'destination_port': np.random.choice(ports),
                        'cost_per_container': np.random.normal(2000, 500),
                        'delay_risk': np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1]),
                        'weight_tons': np.random.normal(15, 3)
                    })
            
            st.session_state.sample_data = pd.DataFrame(data)
        
        return st.session_state.sample_data
    
    def call_perplexity_api(self, query):
        """Call Perplexity AI API for logistics-related queries"""
        api_key = st.secrets.get("PERPLEXITY_API_KEY", "")
        
        if not api_key:
            return "⚠️ Perplexity API key not configured. Please add PERPLEXITY_API_KEY to your Streamlit secrets."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # Enhanced system prompt for logistics context
        system_prompt = """You are an AI assistant specializing in logistics, supply chain management, and shipping operations. 
        Provide accurate, helpful information about:
        - Shipping routes and transportation
        - Container logistics and port operations
        - Supply chain optimization
        - Trade regulations and customs
        - Demand forecasting and inventory management
        - Risk management in logistics
        
        Keep responses concise but informative, and provide actionable insights when possible."""
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.RequestException as e:
            return f"❌ Error calling Perplexity API: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"❌ Error parsing API response: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    
    def create_forecast_model(self, data):
        """Create Prophet forecasting model for shipment demand"""
        if not PROPHET_AVAILABLE:
            return None, "Prophet library not available"
        
        try:
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            daily_demand = data.groupby('date')['shipment_demand'].sum().reset_index()
            daily_demand.columns = ['ds', 'y']
            
            # Create and fit the model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(daily_demand)
            
            # Create future dataframe for next 30 days
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            return model, forecast
        
        except Exception as e:
            return None, f"Error creating forecast: {str(e)}"
    
    def render_chat_tab(self):
        """Render the Chat Assistant tab"""
        st.markdown('<h2 class="tab-header">🤖 AI Logistics Assistant</h2>', unsafe_allow_html=True)
        
        # Chat interface
        st.markdown("Ask me anything about logistics, shipping, supply chain management, or trade operations!")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your logistics question here..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.call_perplexity_api(prompt)
                st.write(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Sidebar with suggested questions
        with st.sidebar:
            st.markdown("### 💡 Suggested Questions")
            suggested_questions = [
                "What are the current shipping rates between Asia and Europe?",
                "How do port congestions affect delivery times?",
                "What factors influence container shipping costs?",
                "How can I optimize my supply chain for seasonal demand?",
                "What are the main trade routes for container shipping?",
                "How do weather conditions impact shipping schedules?"
            ]
            
            for question in suggested_questions:
                if st.button(question, key=f"suggestion_{hash(question)}"):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    response = self.call_perplexity_api(question)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
    
    def render_forecast_tab(self):
        """Render the Demand Forecasting tab"""
        st.markdown('<h2 class="tab-header">📈 Demand Forecasting</h2>', unsafe_allow_html=True)
        
        if not PROPHET_AVAILABLE:
            st.error("Prophet library not available. Please install it to use forecasting features.")
            return
        
        data = st.session_state.sample_data
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### Forecasting Controls")
            
            # Forecasting parameters
            forecast_days = st.selectbox("Forecast Period", [7, 14, 30, 60, 90], index=2)
            route_filter = st.selectbox("Route", ["All Routes"] + list(data['route'].unique()))
            
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Creating forecast model..."):
                    # Filter data if specific route selected
                    forecast_data = data if route_filter == "All Routes" else data[data['route'] == route_filter]
                    
                    # Create forecast
                    model, forecast = self.create_forecast_model(forecast_data)
                    
                    if model is not None:
                        st.session_state.forecast_data = {
                            'model': model,
                            'forecast': forecast,
                            'route': route_filter,
                            'days': forecast_days
                        }
                        st.success("Forecast generated successfully!")
                    else:
                        st.error(f"Failed to generate forecast: {forecast}")
        
        with col1:
            if st.session_state.forecast_data is not None:
                forecast_info = st.session_state.forecast_data
                forecast_df = forecast_info['forecast']
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Historical data
                historical = forecast_df[forecast_df['ds'] <= datetime.now()]
                fig.add_trace(go.Scatter(
                    x=historical['ds'],
                    y=historical['yhat'],
                    mode='lines',
                    name='Historical Trend',
                    line=dict(color='blue')
                ))
                
                # Future forecast
                future = forecast_df[forecast_df['ds'] > datetime.now()]
                fig.add_trace(go.Scatter(
                    x=future['ds'],
                    y=future['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='Confidence Interval',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                
                fig.update_layout(
                    title=f'Shipment Demand Forecast - {forecast_info["route"]}',
                    xaxis_title='Date',
                    yaxis_title='Shipment Demand',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast metrics
                st.markdown("### Forecast Insights")
                future_data = forecast_df[forecast_df['ds'] > datetime.now()].head(forecast_info['days'])
                
                col1_metrics, col2_metrics, col3_metrics = st.columns(3)
                
                with col1_metrics:
                    avg_forecast = future_data['yhat'].mean()
                    st.metric("Avg Daily Demand", f"{avg_forecast:,.0f}", delta=None)
                
                with col2_metrics:
                    total_forecast = future_data['yhat'].sum()
                    st.metric("Total Period Demand", f"{total_forecast:,.0f}", delta=None)
                
                with col3_metrics:
                    max_demand = future_data['yhat'].max()
                    st.metric("Peak Demand Day", f"{max_demand:,.0f}", delta=None)
            
            else:
                st.info("👆 Generate a forecast using the controls on the right to see predictions here.")
    
    def render_dashboard_tab(self):
        """Render the Analytics Dashboard tab"""
        st.markdown('<h2 class="tab-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        data = st.session_state.sample_data
        
        # Key metrics row
        st.markdown("### Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_shipments = len(data)
            st.metric("Total Shipments", f"{total_shipments:,}")
        
        with col2:
            avg_demand = data['shipment_demand'].mean()
            st.metric("Avg Daily Demand", f"{avg_demand:,.0f}")
        
        with col3:
            total_cost = data['cost_per_container'].sum()
            st.metric("Total Cost", f"${total_cost:,.0f}")
        
        with col4:
            high_risk_pct = (data['delay_risk'] == 'High').sum() / len(data) * 100
            st.metric("High Risk Shipments", f"{high_risk_pct:.1f}%")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily demand trend
            daily_demand = data.groupby('date')['shipment_demand'].sum().reset_index()
            fig1 = px.line(
                daily_demand, 
                x='date', 
                y='shipment_demand',
                title='Daily Shipment Demand Trend',
                labels={'shipment_demand': 'Total Daily Demand', 'date': 'Date'}
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Route distribution
            route_demand = data.groupby('route')['shipment_demand'].sum().sort_values(ascending=True)
            fig2 = px.bar(
                x=route_demand.values,
                y=route_demand.index,
                orientation='h',
                title='Demand by Trade Route',
                labels={'x': 'Total Demand', 'y': 'Route'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            # Container type distribution
            container_dist = data['container_type'].value_counts()
            fig3 = px.pie(
                values=container_dist.values,
                names=container_dist.index,
                title='Container Type Distribution'
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Risk analysis
            risk_by_route = data.groupby(['route', 'delay_risk']).size().unstack(fill_value=0)
            fig4 = px.bar(
                risk_by_route,
                title='Risk Assessment by Route',
                labels={'value': 'Number of Shipments', 'index': 'Route'},
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Seasonal analysis
        st.markdown("### Seasonal Analysis")
        
        # Add month column for seasonal analysis
        data['month'] = data['date'].dt.month_name()
        monthly_demand = data.groupby('month')['shipment_demand'].mean().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        
        fig5 = px.bar(
            x=monthly_demand.index,
            y=monthly_demand.values,
            title='Average Monthly Demand Pattern',
            labels={'x': 'Month', 'y': 'Average Demand'}
        )
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Data table
        st.markdown("### Recent Shipment Data")
        display_data = data.head(100)[['date', 'route', 'container_type', 'shipment_demand', 
                                      'origin_port', 'destination_port', 'delay_risk']].copy()
        display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_data, use_container_width=True)

    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🚢 AI-Powered Logistics Assistant</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📈 Demand Forecasting", "📊 Analytics Dashboard"])
        
        with tab1:
            self.render_chat_tab()
        
        with tab2:
            self.render_forecast_tab()
        
        with tab3:
            self.render_dashboard_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**AI-Powered Logistics Assistant** | Built with Streamlit, Prophet, and Perplexity AI | "
            f"Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

# Run the application
if __name__ == "__main__":
    app = LogisticsAssistant()
    app.run()
