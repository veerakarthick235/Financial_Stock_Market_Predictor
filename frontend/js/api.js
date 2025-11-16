const API_BASE_URL = 'http://127.0.0.1:5000';

class StockAPI {
    static async predict(symbol, retrain = false) {
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symbol, retrain })
            });
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }
    
    static async getHistoricalData(symbol, period = '1y') {
        try {
            const response = await fetch(`${API_BASE_URL}/historical`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symbol, period })
            });
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Historical data error:', error);
            throw error;
        }
    }
    
    static async trainModel(symbol) {
        try {
            const response = await fetch(`${API_BASE_URL}/train`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symbol })
            });
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Training error:', error);
            throw error;
        }
    }
    
    static async listModels() {
        try {
            const response = await fetch(`${API_BASE_URL}/models`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('List models error:', error);
            throw error;
        }
    }
}