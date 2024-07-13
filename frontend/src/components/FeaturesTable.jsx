import React from 'react';

const formatDate = (date) => {
    const d = new Date(date);
    let day = d.getDate();
    let month = d.getMonth() + 1;
    day = day < 10 ? `0${day}` : day;
    month = month < 10 ? `0${month}` : month;
    return `${day}/${month}`;
};

const FeaturesTable = ({ features, dates }) => {
    if (!features || Object.keys(features).length === 0) {
        return <div>No data available</div>;
    }

    // Reverse the dates array to show the most recent date first
    const reversedDates = [...dates].reverse();
    const formattedDates = reversedDates.map(formatDate);

    return (
        <div className="table-container">
            <table>
                <thead>
                    <tr>
                        <th className="sticky-header sticky-col">Feature</th>
                        {formattedDates.map(date => <th className="sticky-header" key={date}>{date}</th>)}
                    </tr>
                </thead>
                <tbody>
                    {Object.entries(features).map(([feature, values]) => (
                        // Reverse the values array to match the order of reversedDates
                        <tr key={feature}>
                            <td className="sticky-col">{feature}</td>
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