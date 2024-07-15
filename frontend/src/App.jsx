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
    const [cyclistForTable, setCyclistForTable] = useState('');

    useEffect(() => {
        const fetchCyclistIds = async () => {
            const response = await fetch('https://cyclist-injury-prediction-dc9a079e7f4c.herokuapp.com/cyclists');
            const data = await response.json();
            setCyclistIds(data);
        };
        fetchCyclistIds();
    }, []);

    const handleAddCyclist = async (event) => {
        event.preventDefault();
        if (!selectedCyclists.includes(currentCyclistId)) {
            setSelectedCyclists([...selectedCyclists, currentCyclistId]);
            const response = await fetch('https://cyclist-injury-prediction-dc9a079e7f4c.herokuapp.com/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cyclist_id: Number(currentCyclistId) }),
            });
            const data = await response.json();
            setProbabilities((prev) => ({ ...prev, [currentCyclistId]: data.probabilities }));
            if (!cyclistForTable) {
                setFeatures(data.features);
                setDates(data.dates);
                setCyclistForTable(currentCyclistId);
            }
        }
    };

    const handleRemoveCyclist = (id) => {
        setSelectedCyclists(selectedCyclists.filter((cyclistId) => cyclistId !== id));
        setProbabilities((prev) => {
            const { [id]: _, ...rest } = prev;
            return rest;
        });
        if (cyclistForTable === id) {
            setFeatures({});
            setDates([]);
            setCyclistForTable('');
        }
    };

    const handleCurrentCyclistChange = (event) => {
        setCurrentCyclistId(event.target.value);
    };

    const handleCyclistForTableChange = async (event) => {
        const cyclistId = event.target.value;
        setCyclistForTable(cyclistId);
        const response = await fetch('https://cyclist-injury-prediction-dc9a079e7f4c.herokuapp.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cyclist_id: Number(cyclistId) }),
        });
        const data = await response.json();
        setFeatures(data.features);
        setDates(data.dates);
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
                    <ProbabilityChart data={probabilities} dates={dates} />
                )}
            </div>
            <div className="table-selection">
                {selectedCyclists.length > 0 && (
                    <select
                        value={cyclistForTable}
                        onChange={handleCyclistForTableChange}
                        required
                    >
                        <option value="" disabled>Select Cyclist for Table</option>
                        {selectedCyclists.map((id) => (
                            <option key={id} value={id}>
                                Cyclist ID {id}
                            </option>
                        ))}
                    </select>
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
