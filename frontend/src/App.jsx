import React, { useState, useEffect } from 'react';
import ProbabilityChart from './components/ProbabilityChart';
import FeaturesTable from './components/FeaturesTable';
import './App.css';

const App = () => {
    const [cyclistIds, setCyclistIds] = useState([]);
    const [selectedCyclistId, setSelectedCyclistId] = useState('');
    const [probabilities, setProbabilities] = useState([]);
    const [features, setFeatures] = useState({});
    const [dates, setDates] = useState([]);

    useEffect(() => {
        const fetchCyclistIds = async () => {
            const response = await fetch('http://127.0.0.1:5000/cyclists');
            const data = await response.json();
            setCyclistIds(data);
        };
        fetchCyclistIds();
    }, []);

    const handleSubmit = async (event) => {
        event.preventDefault();
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cyclist_id: Number(selectedCyclistId) }),
        });
        const data = await response.json();
        setProbabilities(data.probabilities);
        setFeatures(data.features);
        setDates(data.dates);
    };

    return (
        <div className="App">
            <h1>Cyclist Injury Probability</h1>
            <form onSubmit={handleSubmit}>
                <select
                    value={selectedCyclistId}
                    onChange={(e) => setSelectedCyclistId(e.target.value)}
                    required
                >
                    <option value="" disabled>Select Cyclist ID</option>
                    {cyclistIds.map((id) => (
                        <option key={id} value={id}>
                            Cyclist ID {id}
                        </option>
                    ))}
                </select>
                <button type="submit">Get Probabilities</button>
            </form>
            <div className="container">
                {probabilities.length > 0 && (
                    <div className="chart">
                        <ProbabilityChart data={probabilities} />
                    </div>
                )}
                {Object.keys(features).length > 0 && (
                    <div className="table">
                        <FeaturesTable features={features} dates={dates} />
                    </div>
                )}
            </div>
        </div>
    );
};

export default App;
