import React from 'react';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';

const ProbabilityChart = ({ data }) => {
    const options = {
        title: {
            text: 'Injury Probability Over 30 Days'
        },
        xAxis: {
            categories: Array.from({ length: 30 }, (_, i) => i + 1),
            title: {
                text: 'Day'
            }
        },
        yAxis: {
            min: 0,
            max: 1,
            title: {
                text: 'Probability'
            }
        },
        series: [{
            name: 'Probability',
            data: data
        }]
    };

    return <HighchartsReact highcharts={Highcharts} options={options} />;
};

export default ProbabilityChart;
