import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import warnings
import os
import google.generativeai as genai
warnings.filterwarnings('ignore')

# Try to import Prophet, handle if not installed
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    st.warning("Prophet not installed. Please install it using: pip install prophet")
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
        font-weight: bold;
    }
    .tab-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class LogisticsAssistant:
    """Main class for the AI-Powered Logistics Assistant"""
    
    def __init__(self):
        self.initialize_session_state()
        self.configure_gemini_api()
        if st.session_state.sample_data is None:
            with st.spinner("Initializing sample data..."):
                self.generate_sample_data()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'sample_data' not in st.session_state:
            st.session_state.sample_data = None
        if 'forecast_data' not in st.session_state:
            st.session_state.forecast_data = None
    
    def get_api_key(self):
        """Get API key from secrets or environment variables"""
        try:
            # Try to get from Streamlit secrets first
            if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
        
        # Fall back to environment variable
        return os.getenv("GEMINI_API_KEY", "")
    
    def configure_gemini_api(self):
        """Configure Google Gemini API"""
        api_key = self.get_api_key()
        if api_key:
            genai.configure(api_key=api_key)
            return True
        return False
    
    @st.cache_data
    def generate_sample_data(_self):
        """Generate sample shipping/logistics data for demonstration"""
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
        ports = ['Shanghai', 'Singapore', 'Rotterdam', 'Los Angeles', 'Hamburg', 'Dubai', 'Hong Kong']
        
        data = []
        for i, date in enumerate(dates):
            for _ in range(np.random.randint(5, 15)):  # 5-15 shipments per day
                origin_port = np.random.choice(ports)
                destination_port = np.random.choice([p for p in ports if p != origin_port])
                
                data.append({
                    'date': date,
                    'shipment_demand': max(50, int(shipment_demand[i] + np.random.normal(0, 100))),
                    'route': np.random.choice(routes),
                    'container_type': np.random.choice(container_types),
                    'origin_port': origin_port,
                    'destination_port': destination_port,
                    'cost_per_container': max(500, np.random.normal(2000, 500)),
                    'delay_risk': np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1]),
                    'weight_tons': max(1, np.random.normal(15, 3))
                })
        
        st.session_state.sample_data = pd.DataFrame(data)
        return st.session_state.sample_data
    
    def call_gemini_api(self, query):
        """Call Google Gemini API for logistics-related queries"""
        api_key = self.get_api_key()
        
        if not api_key:
            return """⚠️ **Google Gemini API key not configured.**
            
Please add your API key in one of these ways:
1. Create `.streamlit/secrets.toml` with: `GEMINI_API_KEY = "your_key"`
2. Set environment variable: `GEMINI_API_KEY=your_key`

Get your API key from: https://makersuite.google.com/app/apikey"""
        
        try:
            # Configure the API key
            genai.configure(api_key=api_key)
            
            # Create the model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Enhanced system prompt for logistics context
            system_prompt = """You are an AI assistant specializing in logistics, supply chain management, and shipping operations. 

Provide accurate, helpful information about:
- Shipping routes and transportation methods
- Container logistics and port operations  
- Supply chain optimization strategies
- Trade regulations and customs procedures
- Demand forecasting and inventory management
- Risk management in logistics operations
- Sustainability in shipping and logistics

Keep responses concise but informative, provide actionable insights, and include relevant data or examples when possible."""
            
            # Combine system prompt with user query
            full_prompt = f"{system_prompt}\n\nUser Question: {query}"
            
            # Generate response
            response = model.generate_content(full_prompt)
            
            return response.text
            
        except Exception as e:
            if "API_KEY_INVALID" in str(e):
                return "❌ **Invalid API Key:** Please check your Google Gemini API key."
            elif "quota" in str(e).lower():
                return "❌ **Quota Exceeded:** You've reached your API usage limit. Please try again later."
            elif "permission" in str(e).lower():
                return "❌ **Permission Denied:** Your API key doesn't have access to Gemini models."
            else:
                return f"❌ **Error calling Gemini API:** {str(e)}"
    
    @st.cache_data
    def create_forecast_model(_self, data, route_filter="All Routes"):
        """Create Prophet forecasting model for shipment demand"""
        if not PROPHET_AVAILABLE:
            return None, "Prophet library not available"
        
        try:
            # Filter data if specific route selected
            if route_filter != "All Routes":
                data = data[data['route'] == route_filter]
            
            if len(data) < 10:
                return None, "Insufficient data for forecasting (minimum 10 data points required)"
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            daily_demand = data.groupby('date')['shipment_demand'].sum().reset_index()
            daily_demand.columns = ['ds', 'y']
            daily_demand = daily_demand.sort_values('ds').reset_index(drop=True)
            
            # Create and fit the model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(daily_demand)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=90)  # 3 months ahead
            forecast = model.predict(future)
            
            return model, forecast
        
        except Exception as e:
            return None, f"Error creating forecast: {str(e)}"
    
    def render_chat_tab(self):
        """Render the Chat Assistant tab"""
        st.markdown('<h2 class="tab-header">🤖 AI Logistics Assistant (Powered by Gemini)</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>💡 Ask me anything about:</strong><br>
            • Shipping routes and logistics optimization<br>
            • Container management and port operations<br>
            • Supply chain risk assessment<br>
            • Trade regulations and customs<br>
            • Demand forecasting strategies
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your logistics question here...", key="main_chat"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your question..."):
                    response = self.call_gemini_api(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Sidebar with suggested questions
        with st.sidebar:
            st.markdown("### 💡 Quick Questions")
            suggested_questions = [
                "What are current shipping delays between major ports?",
                "How do fuel prices affect container shipping costs?",
                "Best practices for cold chain logistics",
                "Impact of weather on shipping schedules",
                "Sustainable shipping solutions",
                "Digital transformation in logistics"
            ]
            
            for i, question in enumerate(suggested_questions):
                if st.button(question, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    response = self.call_gemini_api(question)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    def render_forecast_tab(self):
        """Render the Demand Forecasting tab"""
        st.markdown('<h2 class="tab-header">📈 Demand Forecasting</h2>', unsafe_allow_html=True)
        
        if not PROPHET_AVAILABLE:
            st.error("**Prophet library not available.** Please install it to use forecasting features.")
            st.code("pip install prophet", language="bash")
            return
        
        data = st.session_state.sample_data
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Forecasting Controls")
            
            # Forecasting parameters
            forecast_days = st.selectbox("Forecast Period", [7, 14, 30, 60, 90], index=2)
            route_filter = st.selectbox("Route Filter", ["All Routes"] + sorted(list(data['route'].unique())))
            
            if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
                with st.spinner("Building forecast model..."):
                    model, forecast = self.create_forecast_model(data, route_filter)
                    
                    if model is not None:
                        st.session_state.forecast_data = {
                            'model': model,
                            'forecast': forecast,
                            'route': route_filter,
                            'days': forecast_days
                        }
                        st.success("✅ Forecast generated successfully!")
                    else:
                        st.error(f"❌ Failed to generate forecast: {forecast}")
        
        with col1:
            if st.session_state.forecast_data is not None:
                forecast_info = st.session_state.forecast_data
                forecast_df = forecast_info['forecast']
                
                # Create forecast visualization
                fig = go.Figure()
                
                # Split historical and future data
                current_date = datetime.now()
                historical = forecast_df[forecast_df['ds'] <= current_date]
                future = forecast_df[forecast_df['ds'] > current_date].head(forecast_info['days'])
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical['ds'],
                    y=historical['yhat'],
                    mode='lines',
                    name='Historical Trend',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Future forecast
                fig.add_trace(go.Scatter(
                    x=future['ds'],
                    y=future['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=3, dash='dash')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=future['ds'],
                    y=future['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future['ds'],
                    y=future['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='80% Confidence',
                    fillcolor='rgba(255,127,14,0.2)'
                ))
                
                fig.update_layout(
                    title=f'📊 Shipment Demand Forecast - {forecast_info["route"]}',
                    xaxis_title='Date',
                    yaxis_title='Daily Shipment Demand',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast metrics
                st.markdown("### 📊 Forecast Insights")
                
                if len(future) > 0:
                    col1_metrics, col2_metrics, col3_metrics, col4_metrics = st.columns(4)
                    
                    with col1_metrics:
                        avg_forecast = future['yhat'].mean()
                        st.metric("📈 Avg Daily Demand", f"{avg_forecast:,.0f}")
                    
                    with col2_metrics:
                        total_forecast = future['yhat'].sum()
                        st.metric("📦 Total Period Demand", f"{total_forecast:,.0f}")
                    
                    with col3_metrics:
                        max_demand = future['yhat'].max()
                        peak_date = future.loc[future['yhat'].idxmax(), 'ds'].strftime('%m/%d')
                        st.metric("⚡ Peak Demand", f"{max_demand:,.0f}", delta=f"on {peak_date}")
                    
                    with col4_metrics:
                        # ✅ FIXED: Added [0] index to compare first and last values
                        trend = "📈 Growing" if future['yhat'].iloc[-1] > future['yhat'].iloc[0] else "📉 Declining"
                        change_pct = ((future['yhat'].iloc[-1] - future['yhat'].iloc[0]) / future['yhat'].iloc[0]) * 100
                        st.metric("📊 Trend", trend, delta=f"{change_pct:+.1f}%")
            
            else:
                st.info("👆 **Generate a forecast** using the controls to see predictions and insights here.")
                
                # Show sample historical data chart
                daily_demand = data.groupby('date')['shipment_demand'].sum().reset_index()
                fig_sample = px.line(daily_demand.tail(90), x='date', y='shipment_demand',
                                   title='📊 Recent 90 Days - Historical Demand')
                fig_sample.update_layout(height=400)
                st.plotly_chart(fig_sample, use_container_width=True)
    
    def render_dashboard_tab(self):
        """Render the Analytics Dashboard tab"""
        st.markdown('<h2 class="tab-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        data = st.session_state.sample_data
        
        # Date range selector
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            start_date = st.date_input("Start Date", data['date'].min().date())
        with col2:
            end_date = st.date_input("End Date", data['date'].max().date())
        with col3:
            refresh_data = st.button("🔄 Refresh", use_container_width=True)
        
        # Filter data by date range
        mask = (data['date'].dt.date >= start_date) & (data['date'].dt.date <= end_date)
        filtered_data = data.loc[mask]
        
        # Key metrics row
        st.markdown("### 🎯 Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_shipments = len(filtered_data)
            st.metric("🚢 Total Shipments", f"{total_shipments:,}")
        
        with col2:
            avg_demand = filtered_data['shipment_demand'].mean()
            st.metric("📊 Avg Demand", f"{avg_demand:,.0f}")
        
        with col3:
            total_cost = filtered_data['cost_per_container'].sum()
            st.metric("💰 Total Cost", f"${total_cost:,.0f}")
        
        with col4:
            high_risk_pct = (filtered_data['delay_risk'] == 'High').sum() / len(filtered_data) * 100
            st.metric("⚠️ High Risk %", f"{high_risk_pct:.1f}%")
        
        with col5:
            unique_routes = filtered_data['route'].nunique()
            st.metric("🌍 Active Routes", f"{unique_routes}")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily demand trend
            daily_demand = filtered_data.groupby('date')['shipment_demand'].sum().reset_index()
            fig1 = px.line(
                daily_demand, 
                x='date', 
                y='shipment_demand',
                title='📈 Daily Shipment Demand Trend',
                labels={'shipment_demand': 'Total Daily Demand', 'date': 'Date'}
            )
            fig1.update_traces(line_color='#1f77b4')
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Route performance
            route_stats = filtered_data.groupby('route').agg({
                'shipment_demand': 'sum',
                'cost_per_container': 'mean'
            }).round(0)
            
            fig2 = px.bar(
                x=route_stats['shipment_demand'].values,
                y=route_stats.index,
                orientation='h',
                title='🌍 Demand by Trade Route',
                labels={'x': 'Total Demand', 'y': 'Route'},
                color=route_stats['shipment_demand'].values,
                color_continuous_scale='Blues'
            )
            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            # Container type distribution
            container_dist = filtered_data['container_type'].value_counts()
            fig3 = px.pie(
                values=container_dist.values,
                names=container_dist.index,
                title='📦 Container Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig3.update_traces(textposition='inside', textinfo='percent+label')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Risk heatmap by route
            risk_pivot = filtered_data.groupby(['route', 'delay_risk']).size().unstack(fill_value=0)
            
            fig4 = px.imshow(
                risk_pivot.values,
                labels=dict(x="Risk Level", y="Route", color="Count"),
                x=risk_pivot.columns,
                y=risk_pivot.index,
                title='🎯 Risk Assessment Heatmap',
                color_continuous_scale='RdYlGn_r'
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Advanced analytics
        st.markdown("### 📊 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seasonal analysis
            filtered_data['month'] = filtered_data['date'].dt.month_name()
            monthly_demand = filtered_data.groupby('month')['shipment_demand'].mean().reindex([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])
            
            fig5 = px.bar(
                x=monthly_demand.index,
                y=monthly_demand.values,
                title='📅 Average Monthly Demand Pattern',
                labels={'x': 'Month', 'y': 'Average Demand'},
                color=monthly_demand.values,
                color_continuous_scale='Viridis'
            )
            fig5.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # Cost vs Demand scatter
            route_analysis = filtered_data.groupby('route').agg({
                'shipment_demand': 'mean',
                'cost_per_container': 'mean',
                'delay_risk': lambda x: (x == 'High').sum() / len(x) * 100
            }).round(2)
            
            fig6 = px.scatter(
                route_analysis,
                x='cost_per_container',
                y='shipment_demand',
                size='delay_risk',
                title='💰 Cost vs Demand Analysis',
                labels={'cost_per_container': 'Avg Cost per Container ($)', 
                       'shipment_demand': 'Avg Demand'},
                hover_data={'delay_risk': ':.1f'}
            )
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
        
        # Data summary table
        st.markdown("### 📋 Recent Shipment Data")
        
        # Add filters for the table
        col1, col2, col3 = st.columns(3)
        with col1:
            route_filter = st.selectbox("Filter by Route", ["All"] + sorted(list(filtered_data['route'].unique())))
        with col2:
            risk_filter = st.selectbox("Filter by Risk", ["All"] + sorted(list(filtered_data['delay_risk'].unique())))
        with col3:
            container_filter = st.selectbox("Filter by Container", ["All"] + sorted(list(filtered_data['container_type'].unique())))
        
        # Apply table filters
        table_data = filtered_data.copy()
        if route_filter != "All":
            table_data = table_data[table_data['route'] == route_filter]
        if risk_filter != "All":
            table_data = table_data[table_data['delay_risk'] == risk_filter]
        if container_filter != "All":
            table_data = table_data[table_data['container_type'] == container_filter]
        
        # Display filtered data
        display_data = table_data.head(200)[['date', 'route', 'container_type', 'shipment_demand', 
                                           'origin_port', 'destination_port', 'delay_risk', 'cost_per_container']].copy()
        display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d')
        display_data['cost_per_container'] = display_data['cost_per_container'].round(2)
        
        st.dataframe(
            display_data, 
            use_container_width=True,
            column_config={
                "cost_per_container": st.column_config.NumberColumn(
                    "Cost per Container ($)",
                    format="$%.2f"
                )
            }
        )
        
        # Download button
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"logistics_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🚢 AI-Powered Logistics Assistant</h1>', unsafe_allow_html=True)
        st.markdown("**Powered by Google Gemini AI**")
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "💬 Chat Assistant", 
            "📈 Demand Forecasting", 
            "📊 Analytics Dashboard"
        ])
        
        with tab1:
            self.render_chat_tab()
        
        with tab2:
            self.render_forecast_tab()
        
        with tab3:
            self.render_dashboard_tab()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🚢 AI-Powered Logistics Assistant**")
        
        with col2:
            st.markdown("Built with Streamlit • Prophet • Google Gemini")
        
        with col3:
            st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Run the application
if __name__ == "__main__":
    app = LogisticsAssistant()
    app.run()
