import React from 'react';

const FeaturesTable = ({ features, dates }) => {
    if (!features || Object.keys(features).length === 0) {
        return <div>No data available</div>;
    }

    // Reverse the dates array to show the most recent date first
    const reversedDates = [...dates].reverse();

    return (
        <div>
            <table>
                <thead>
                <tr>
                    <th>Feature</th>
                    {reversedDates.map(date => <th key={date}>{date}</th>)}
                </tr>
                </thead>
                <tbody>
                {Object.entries(features).map(([feature, values]) => (
                    // Reverse the values array to match the order of reversedDates
                    <tr key={feature}>
                        <td>{feature}</td>
                        {[...values].reverse().map((value, index) => (
                            <td key={index}>{value}</td>
                        ))}
                    </tr>
                ))}
                </tbody>
            </table>
        </div>
    );
};

export default FeaturesTable;
