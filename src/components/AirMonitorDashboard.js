import React, { useEffect, useState } from "react";
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";
import './AirMonitor.css'; // Import the CSS file

const API_KEY = process.env.REACT_APP_AIR_MONITOR_API_KEY; 
const CITY = 'Atlanta';

const AirMonitorDashboard = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://api.weatherapi.com/v1/forecast.json?key=' + API_KEY  + '&q=' + CITY + '&aqi=yes');
                if (!response.ok) {
                    throw new Error(`HTTP Error: ${response.status}`);
                }
                const result = await response.json();
                setData(result.current.air_quality);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    if (loading) return <p className="text-center text-lg">Loading air quality data...</p>;
    if (error) return <p className="text-center text-red-500">Error: {error}</p>;

    const { co, no2, o3, pm10, pm2_5, so2, "us-epa-index": aqi } = data;

    const graphData = [
        { name: "CO₂ (CO)", value: co },
        { name: "NO₂", value: no2 },
        { name: "O₃", value: o3 },
        { name: "PM10", value: pm10 },
        { name: "PM2.5", value: pm2_5 },
        { name: "SO₂", value: so2 },
    ];

    const gasLevels = [
        { gas: "CO", value: co, description: "Carbon Monoxide (CO) is a colorless, odorless gas produced by combustion. High levels can impair oxygen delivery in the body." },
        { gas: "NO₂", value: no2, description: "Nitrogen Dioxide (NO₂) is a key pollutant in urban areas, produced by combustion processes like vehicles and industrial activities." },
        { gas: "O₃", value: o3, description: "Ozone (O₃) is a reactive gas in the atmosphere that can cause respiratory issues at high levels." },
        { gas: "PM10", value: pm10, description: "Particulate Matter (PM10) consists of airborne particles that can irritate the lungs and heart." },
        { gas: "PM2.5", value: pm2_5, description: "PM2.5 particles are fine inhalable particles that pose a significant health risk, especially for the respiratory system." },
        { gas: "SO₂", value: so2, description: "Sulfur Dioxide (SO₂) is produced by industrial processes and can cause respiratory problems when inhaled." },
        { gas: "AQI", value: aqi, description: "Air Quality Index (AQI) is a measure of air quality, indicating how polluted the air currently is or how polluted it is forecast to become." },
        
    ];

    return (
        <div className="dashboard-container">
            <h1 className="dashboard-title">Air Quality Dashboard</h1>

            {/* Air Quality Summary */}
            <div className="summary-container">
                {gasLevels.map((gas) => (
                    <div className="summary-card" key={gas.gas}>
                        <h3 className="card-title">{gas.gas}</h3>
                        <p className="card-value">{gas.value.toFixed(2)} µg/m³</p>
                        <p className="card-description">{gas.description}</p>
                    </div>
                ))}
            </div>

            {/* Scrollable Charts Section */}
            <div className="chart-section">
                {/* AQI LineChart */}
                <h2 className="chart-title">Air Quality Index (AQI) Trends</h2>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={graphData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <ReferenceLine y={50} label="Good AQI" stroke="green" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="value" stroke="#1D4ED8" dot={false} activeDot={{ r: 8 }} />
                    </LineChart>
                </ResponsiveContainer>

                {/* BarChart for Detailed Data */}
                <h2 className="chart-title">Detailed Air Quality Data</h2>
                <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={graphData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="value" fill="#FF6F61" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default AirMonitorDashboard;
