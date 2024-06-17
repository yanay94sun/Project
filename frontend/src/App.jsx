import React, { useState, useEffect } from 'react';
import ProbabilityChart from './components/ProbabilityChart';
import FeaturesTable from './components/FeaturesTable';
import './App.css';

const App = () => {
    const [cyclistIds, setCyclistIds] = useState([]);
    const [selectedCyclists, setSelectedCyclists] = useState([]);
    const [currentCyclistId, setCurrentCyclistId] = useState('');
    const [probabilities, setProbabilities] = useState({});
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

    const handleAddCyclist = async (event) => {
        event.preventDefault();
        if (!selectedCyclists.includes(currentCyclistId)) {
            setSelectedCyclists([...selectedCyclists, currentCyclistId]);
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cyclist_id: Number(currentCyclistId) }),
            });
            const data = await response.json();
            setProbabilities((prev) => ({ ...prev, [currentCyclistId]: data.probabilities }));
            setFeatures(data.features);
            setDates(data.dates);
        }
    };

    const handleRemoveCyclist = (id) => {
        setSelectedCyclists(selectedCyclists.filter((cyclistId) => cyclistId !== id));
        setProbabilities((prev) => {
            const { [id]: _, ...rest } = prev;
            return rest;
        });
    };

    const handleCurrentCyclistChange = (event) => {
        setCurrentCyclistId(event.target.value);
    };

    return (
        <div className="App">
            <h1>Cyclist Injury Probability</h1>
            <form onSubmit={handleAddCyclist}>
                <select
                    value={currentCyclistId}
                    onChange={handleCurrentCyclistChange}
                    required
                >
                    <option value="" disabled>Select Cyclist ID</option>
                    {cyclistIds.map((id) => (
                        <option key={id} value={id}>
                            Cyclist ID {id}
                        </option>
                    ))}
                </select>
                <button type="submit">Add Cyclist</button>
            </form>
            <div className="selected-cyclists">
                {selectedCyclists.map((id) => (
                    <div key={id} className="selected-cyclist">
                        <span>Cyclist ID {id}</span>
                        <button onClick={() => handleRemoveCyclist(id)}>Remove</button>
                    </div>
                ))}
            </div>
            <div className="chart">
                {Object.keys(probabilities).length > 0 && (
                    <ProbabilityChart data={probabilities} />
                )}
            </div>
            <div className="table">
                {Object.keys(features).length > 0 && (
                    <FeaturesTable features={features} dates={dates} />
                )}
            </div>
        </div>
    );
};

export default App;
