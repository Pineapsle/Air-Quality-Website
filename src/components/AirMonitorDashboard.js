import React, { useEffect, useState } from "react";
import { PolarGrid, PolarAngleAxis, PolarRadiusAxis, RadarChart, Radar, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from "recharts";
import './AirMonitor.css'; // Import the CSS file

const API_KEY = process.env.REACT_APP_AIR_MONITOR_API_KEY; 
const CITY = 'Atlanta';

const AirMonitorDashboard = () => {
    const [gas_data, setData] = useState(null);
    const [weather_data, setWeatherData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://api.weatherapi.com/v1/forecast.json?key=' + API_KEY  + '&q=' + CITY + '&aqi=yes');
                if (!response.ok) {
                    throw new Error('HTTP Error:' + response.status);
                }
                const result = await response.json();
                setData(result.current.air_quality);
                setWeatherData(result.forecast.forecastday[0].day);
                
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

    const { co, no2, o3, pm10, pm2_5, so2, "us-epa-index": usEpaIndex, "gb-defra-index": gbDefraIndex } = gas_data;
    const { avgtemp_c, avghumidity, daily_chance_of_rain, maxwind_mph } = weather_data;

    const gasGraphData = [
        { name: "CO (CO)", value: co },
        { name: "NO₂", value: no2 },
        { name: "O₃", value: o3 },
        { name: "PM10", value: pm10 },
        { name: "PM2.5", value: pm2_5 },
        { name: "SO₂", value: so2 },
    ];

    const gasDataLevels = [
        { gas_data: "CO", value: co, description: "Carbon Monoxide (CO) is a colorless, odorless gas produced by combustion. High levels can impair oxygen delivery in the body." },
        { gas_data: "NO₂", value: no2, description: "Nitrogen Dioxide (NO₂) is a key pollutant in urban areas, produced by combustion processes like vehicles and industrial activities." },
        { gas_data: "O₃", value: o3, description: "Ozone (O₃) is a reactive gas in the atmosphere that can cause respiratory issues at high levels." },
        { gas_data: "PM10", value: pm10, description: "Particulate Matter (PM10) consists of airborne particles that can irritate the lungs and heart." },
        { gas_data: "PM2.5", value: pm2_5, description: "PM2.5 particles are fine inhalable particles that pose a significant health risk, especially for the respiratory system." },
        { gas_data: "SO₂", value: so2, description: "Sulfur Dioxide (SO₂) is produced by industrial processes and can cause respiratory problems when inhaled." },
        { gas_data: "Environmental Quality Index (EQI)", value: usEpaIndex, description: "Association between environmental quality and health concerns based off factors such as water quality and soil condition." },
        { gas_data: "Daily Air Quality Index (DAQI)", value: gbDefraIndex, description: "The Air Quality Index (AQI) tracks air pollution levels, helping assess health risks from pollutants like PM2.5, NO₂, and O₃." },
    ];

    const weatherDataLevels = [
        { weather_data: "Temperature (C)", value: avgtemp_c, description: "Current temperature in Celsius." },
        { weather_data: "Humidity", value: avghumidity, description: "Current humidity percentage." },
        { weather_data: "Chance of Rain", value: daily_chance_of_rain, description: "Chance of rain." },
        { weather_data: "Wind Speed (MPH)", value: maxwind_mph, description: "Maximum wind speed in miles per hour." },
    ];

    return (
        <div className="dashboard-container">
            <h1 className="dashboard-title">Air Quality</h1>

            {/* Air Quality Summary */}
            <div className="summary-container">
                {gasDataLevels.map((gas) => (
                    <div className="summary-card" key={gas.gas_data}>
                        <h3 className="card-title">{gas.gas_data}</h3>
                        <p className="card-value">{gas.value} µg/m³</p>
                        <p className="card-description">{gas.description}</p>
                    </div>
                ))}
            </div>

            <h1 className="dashboard-title">Weather Data</h1>

            {/* Weather Summary */}
            <div className="summary-container">
                {weatherDataLevels.map((weather) => (
                    <div className="summary-card" key={weather.weather_data}>
                        <h3 className="card-title">{weather.weather_data}</h3>
                        <p className="card-value">{weather.value}</p>
                        <p className="card-description">{weather.description}</p>
                    </div>
                ))}
            </div>

            {/* Scrollable Charts Section */}
            <div className="chart-section">
                {/* AQI LineChart */}
                <h2 className="chart-title">Air Quality Index (AQI) Trends</h2>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={gasGraphData}>
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
                    <BarChart data={gasGraphData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="value" fill="#FF6F61" />
                    </BarChart>
                </ResponsiveContainer>

                {/* PieChart for Detailed Data */}
                <h2 className="chart-title">Detailed Air Quality Data</h2>
                <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={gasGraphData}>
                    <PolarGrid />
                        <PolarAngleAxis dataKey="name" />
                        <PolarRadiusAxis angle={30} domain={[0, 150]} />
                        <Radar name="Gases" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                        <Legend />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default AirMonitorDashboard;
