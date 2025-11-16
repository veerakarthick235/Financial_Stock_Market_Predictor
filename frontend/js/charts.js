class ChartManager {
    static forecastChart = null;
    static historicalChart = null;
    static analysisChart = null;
    static rsiChart = null;
    
    static createForecastChart(predictions) {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        // Destroy existing chart
        if (this.forecastChart) {
            this.forecastChart.destroy();
        }
        
        const labels = predictions.map((_, index) => `Day ${index + 1}`);
        
        this.forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Predicted Price',
                    data: predictions,
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: '#00ff88',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: '#ffffff',
                            font: {
                                family: 'Rajdhani',
                                size: 14
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 14, 39, 0.95)',
                        titleColor: '#00ff88',
                        bodyColor: '#ffffff',
                        borderColor: '#00ff88',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                return '$' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        grid: {
                            color: 'rgba(0, 255, 136, 0.1)'
                        },
                        ticks: {
                            color: '#a0aec0',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 255, 136, 0.1)'
                        },
                        ticks: {
                            color: '#a0aec0'
                        }
                    }
                }
            }
        });
    }
    
    static createHistoricalChart(dates, prices) {
        const ctx = document.getElementById('historicalChart').getContext('2d');
        
        if (this.historicalChart) {
            this.historicalChart.destroy();
        }
        
        this.historicalChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Historical Price',
                    data: prices,
                    borderColor: '#0088ff',
                    backgroundColor: 'rgba(0, 136, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff',
                            font: { family: 'Rajdhani', size: 14 }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 14, 39, 0.95)',
                        titleColor: '#0088ff',
                        bodyColor: '#ffffff',
                        borderColor: '#0088ff',
                        borderWidth: 1
                    }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(0, 136, 255, 0.1)' },
                        ticks: {
                            color: '#a0aec0',
                            callback: (value) => '$' + value.toFixed(2)
                        }
                    },
                    x: {
                        grid: { color: 'rgba(0, 136, 255, 0.1)' },
                        ticks: {
                            color: '#a0aec0',
                            maxTicksLimit: 10
                        }
                    }
                }
            }
        });
    }
    
    static createAnalysisChart(data) {
        const ctx = document.getElementById('analysisChart').getContext('2d');
        
        if (this.analysisChart) {
            this.analysisChart.destroy();
        }
        
        this.analysisChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [
                    {
                        label: 'Close Price',
                        data: data.close,
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        yAxisID: 'y'
                    },
                    {
                        label: 'SMA (20)',
                        data: data.sma_20,
                        borderColor: '#0088ff',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'EMA (20)',
                        data: data.ema_20,
                        borderColor: '#ff0088',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff',
                            font: { family: 'Rajdhani', size: 14 }
                        }
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        position: 'left',
                        grid: { color: 'rgba(0, 255, 136, 0.1)' },
                        ticks: {
                            color: '#a0aec0',
                            callback: (value) => '$' + value.toFixed(2)
                        }
                    },
                    x: {
                        grid: { color: 'rgba(0, 255, 136, 0.1)' },
                        ticks: {
                            color: '#a0aec0',
                            maxTicksLimit: 15
                        }
                    }
                }
            }
        });
    }
    
    static createRSIChart(dates, rsi) {
        const ctx = document.getElementById('rsiChart').getContext('2d');
        
        if (this.rsiChart) {
            this.rsiChart.destroy();
        }
        
        this.rsiChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'RSI',
                    data: rsi,
                    borderColor: '#ffaa00',
                    backgroundColor: 'rgba(255, 170, 0, 0.1)',
                    borderWidth: 2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff',
                            font: { family: 'Rajdhani', size: 14 }
                        }
                    },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: 70,
                                yMax: 70,
                                borderColor: '#ff0055',
                                borderWidth: 2,
                                borderDash: [5, 5]
                            },
                            line2: {
                                type: 'line',
                                yMin: 30,
                                yMax: 30,
                                borderColor: '#00ff88',
                                borderWidth: 2,
                                borderDash: [5, 5]
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255, 170, 0, 0.1)' },
                        ticks: { color: '#a0aec0' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 170, 0, 0.1)' },
                        ticks: {
                            color: '#a0aec0',
                            maxTicksLimit: 15
                        }
                    }
                }
            }
        });
    }
}