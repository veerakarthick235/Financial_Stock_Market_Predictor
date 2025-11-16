// Utility Functions
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function setConfidenceScore(score) {
    const circle = document.getElementById('confidenceCircle');
    const desc = document.getElementById('confidenceDesc');
    
    circle.style.setProperty('--progress', score);
    circle.querySelector('.progress-value').textContent = Math.round(score) + '%';
    
    if (score >= 75) {
        desc.textContent = 'High Confidence';
        desc.style.color = '#00ff88';
    } else if (score >= 50) {
        desc.textContent = 'Medium Confidence';
        desc.style.color = '#ffaa00';
    } else {
        desc.textContent = 'Low Confidence';
        desc.style.color = '#ff0055';
    }
}

function setSentimentGauge(score) {
    const fill = document.getElementById('sentimentFill');
    const value = document.getElementById('sentimentValue');
    
    fill.style.left = `${score}%`;
    value.textContent = Math.round(score);
    
    if (score >= 60) {
        value.style.color = '#00ff88';
    } else if (score >= 40) {
        value.style.color = '#ffaa00';
    } else {
        value.style.color = '#ff0055';
    }
}

function setIndicatorStatus(rsi, macd, sma, currentPrice) {
    // RSI
    const rsiValue = document.getElementById('rsiValue');
    const rsiStatus = document.getElementById('rsiStatus');
    
    if (rsi !== null) {
        rsiValue.textContent = rsi.toFixed(2);
        
        if (rsi > 70) {
            rsiStatus.textContent = 'Overbought';
            rsiStatus.className = 'indicator-status bearish';
        } else if (rsi < 30) {
            rsiStatus.textContent = 'Oversold';
            rsiStatus.className = 'indicator-status bullish';
        } else {
            rsiStatus.textContent = 'Neutral';
            rsiStatus.className = 'indicator-status neutral';
        }
    }
    
    // MACD
    const macdValue = document.getElementById('macdValue');
    const macdStatus = document.getElementById('macdStatus');
    
    if (macd !== null) {
        macdValue.textContent = macd.toFixed(2);
        
        if (macd > 0) {
            macdStatus.textContent = 'Bullish';
            macdStatus.className = 'indicator-status bullish';
        } else {
            macdStatus.textContent = 'Bearish';
            macdStatus.className = 'indicator-status bearish';
        }
    }
    
    // SMA
    const smaValue = document.getElementById('smaValue');
    const smaStatus = document.getElementById('smaStatus');
    
    if (sma !== null) {
        smaValue.textContent = '$' + sma.toFixed(2);
        
        if (currentPrice > sma) {
            smaStatus.textContent = 'Above SMA';
            smaStatus.className = 'indicator-status bullish';
        } else {
            smaStatus.textContent = 'Below SMA';
            smaStatus.className = 'indicator-status bearish';
        }
    }
}

// Navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const section = btn.dataset.section;
        
        // Update active button
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Show section
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById(section).classList.add('active');
    });
});

// Quick Select Chips
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const symbol = chip.dataset.symbol;
        document.getElementById('symbolInput').value = symbol;
    });
});

// Prediction
document.getElementById('predictBtn').addEventListener('click', async () => {
    const symbol = document.getElementById('symbolInput').value.trim();
    
    if (!symbol) {
        showToast('Please enter a stock symbol', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const result = await StockAPI.predict(symbol);
        
        if (result.success) {
            // Show results container
            document.getElementById('resultsContainer').style.display = 'grid';
            
            // Stock Info
            document.getElementById('stockName').textContent = result.stock_info.name;
            document.getElementById('stockSymbol').textContent = result.symbol;
            document.getElementById('stockSector').textContent = result.stock_info.sector;
            document.getElementById('stockExchange').textContent = result.stock_info.exchange;
            document.getElementById('stockCurrency').textContent = result.stock_info.currency;
            
            // Prices
            document.getElementById('currentPrice').textContent = '$' + result.current_price.toFixed(2);
            document.getElementById('predictedPrice').textContent = '$' + result.predicted_price.toFixed(2);
            
            const changeElem = document.getElementById('priceChange');
            const change = result.price_change_percent;
            changeElem.textContent = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
            changeElem.className = change >= 0 ? 'change-value positive' : 'change-value negative';
            
            // Confidence & Sentiment
            setConfidenceScore(result.confidence_score);
            setSentimentGauge(result.sentiment_score);
            
            // Technical Indicators
            setIndicatorStatus(
                result.technical_indicators.rsi,
                result.technical_indicators.macd,
                result.technical_indicators.sma_20,
                result.current_price
            );
            
            // Charts
            ChartManager.createForecastChart(result.week_predictions);
            ChartManager.createHistoricalChart(
                result.historical_data.dates,
                result.historical_data.prices
            );
            
            showToast('Prediction completed successfully!');
        } else {
            showToast(result.error || 'Prediction failed', 'error');
        }
    } catch (error) {
        showToast('Error connecting to server', 'error');
        console.error(error);
    }
    
    hideLoading();
});

// Analysis
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const symbol = document.getElementById('analyzeSymbol').value.trim();
    const period = document.getElementById('periodSelect').value;
    
    if (!symbol) {
        showToast('Please enter a stock symbol', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const result = await StockAPI.getHistoricalData(symbol, period);
        
        if (result.success) {
            document.getElementById('analysisResults').style.display = 'grid';
            
            ChartManager.createAnalysisChart(result);
            ChartManager.createRSIChart(result.dates, result.rsi);
            
            showToast('Analysis completed!');
        } else {
            showToast(result.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        showToast('Error fetching analysis data', 'error');
        console.error(error);
    }
    
    hideLoading();
});

// Training
document.getElementById('trainBtn').addEventListener('click', async () => {
    const symbol = document.getElementById('trainSymbol').value.trim();
    
    if (!symbol) {
        showToast('Please enter a stock symbol', 'error');
        return;
    }
    
    const progressDiv = document.getElementById('trainingProgress');
    const resultsDiv = document.getElementById('trainingResults');
    const statusText = document.getElementById('trainingStatus');
    
    progressDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    statusText.textContent = 'Training in progress...';
    
    showLoading();
    
    try {
        const result = await StockAPI.trainModel(symbol);
        
        if (result.success) {
            statusText.textContent = 'Training completed!';
            
            // Show results
            resultsDiv.style.display = 'block';
            document.getElementById('trainLoss').textContent = result.metrics.train_loss.toFixed(6);
            document.getElementById('valLoss').textContent = result.metrics.val_loss.toFixed(6);
            document.getElementById('epochsCount').textContent = result.metrics.epochs_trained;
            
            showToast('Model trained successfully!');
            loadModelsList();
        } else {
            showToast(result.error || 'Training failed', 'error');
            progressDiv.style.display = 'none';
        }
    } catch (error) {
        showToast('Error training model', 'error');
        progressDiv.style.display = 'none';
        console.error(error);
    }
    
    hideLoading();
});

// Load Models List
async function loadModelsList() {
    try {
        const result = await StockAPI.listModels();
        
        const modelsList = document.getElementById('modelsList');
        
        if (result.success && result.models.length > 0) {
            modelsList.innerHTML = result.models.map(model => `
                <div class="model-item">
                    <i class="fas fa-brain"></i>
                    <span>${model}</span>
                </div>
            `).join('');
        } else {
            modelsList.innerHTML = '<p class="empty-state">No trained models yet</p>';
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Portfolio Management
class Portfolio {
    static get() {
        return JSON.parse(localStorage.getItem('portfolio') || '[]');
    }
    
    static add(item) {
        const portfolio = this.get();
        portfolio.push(item);
        localStorage.setItem('portfolio', JSON.stringify(portfolio));
        this.render();
    }
    
    static remove(index) {
        const portfolio = this.get();
        portfolio.splice(index, 1);
        localStorage.setItem('portfolio', JSON.stringify(portfolio));
        this.render();
    }
    
    static render() {
        const portfolio = this.get();
        const list = document.getElementById('portfolioList');
        
        if (portfolio.length === 0) {
            list.innerHTML = '<p class="empty-state">No stocks in portfolio. Add some to get started!</p>';
            return;
        }
        
        list.innerHTML = portfolio.map((item, index) => {
            const currentValue = item.shares * item.buyPrice; // Simplified
            const gain = 0; // Would need real-time prices
            
            return `
                <div class="portfolio-item">
                    <div class="portfolio-item-field">
                        <span class="portfolio-item-label">Symbol</span>
                        <span class="portfolio-item-value">${item.symbol}</span>
                    </div>
                    <div class="portfolio-item-field">
                        <span class="portfolio-item-label">Shares</span>
                        <span class="portfolio-item-value">${item.shares}</span>
                    </div>
                    <div class="portfolio-item-field">
                        <span class="portfolio-item-label">Buy Price</span>
                        <span class="portfolio-item-value">$${item.buyPrice}</span>
                    </div>
                    <div class="portfolio-item-field">
                        <span class="portfolio-item-label">Value</span>
                        <span class="portfolio-item-value">$${currentValue.toFixed(2)}</span>
                    </div>
                    <div class="portfolio-item-field">
                        <span class="portfolio-item-label">Gain/Loss</span>
                        <span class="portfolio-item-value">$${gain.toFixed(2)}</span>
                    </div>
                    <button class="btn-remove" onclick="Portfolio.remove(${index})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
        }).join('');
    }
}

document.getElementById('addToPortfolioBtn').addEventListener('click', () => {
    const symbol = document.getElementById('portfolioSymbol').value.trim();
    const shares = parseFloat(document.getElementById('portfolioShares').value);
    const buyPrice = parseFloat(document.getElementById('portfolioPrice').value);
    
    if (!symbol || !shares || !buyPrice) {
        showToast('Please fill all fields', 'error');
        return;
    }
    
    Portfolio.add({ symbol, shares, buyPrice });
    
    // Clear inputs
    document.getElementById('portfolioSymbol').value = '';
    document.getElementById('portfolioShares').value = '';
    document.getElementById('portfolioPrice').value = '';
    
    showToast('Stock added to portfolio!');
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModelsList();
    Portfolio.render();
});

// Add CSS for model items
const style = document.createElement('style');
style.textContent = `
    .model-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: rgba(0, 20, 40, 0.3);
        border-radius: 8px;
        border: 1px solid rgba(0, 255, 136, 0.1);
        margin-bottom: 0.5rem;
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .model-item i {
        color: var(--primary-color);
        font-size: 1.2rem;
    }
    
    .training-metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metric {
        text-align: center;
        padding: 1rem;
        background: rgba(0, 20, 40, 0.3);
        border-radius: 8px;
        border: 1px solid rgba(0, 255, 136, 0.1);
    }
    
    .metric-label {
        display: block;
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    
    .metric-value {
        display: block;
        color: var(--primary-color);
        font-family: var(--font-display);
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .progress-text {
        text-align: center;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    .analysis-controls,
    .train-controls,
    .portfolio-controls {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
`;
document.head.appendChild(style);