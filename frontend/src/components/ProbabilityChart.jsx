import React from 'react';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official';

const formatDate = (date) => {
    const d = new Date(date);
    let day = d.getDate();
    let month = d.getMonth() + 1;
    day = day < 10 ? `0${day}` : day;
    month = month < 10 ? `0${month}` : month;
    return `${day}/${month}`;
};

const getNextDates = (dates) => {
    return dates.map((date) => {
        const d = new Date(date);
        d.setDate(d.getDate() + 30); // Add 30 days
        return formatDate(d);
    });
};

const ProbabilityChart = ({ data, dates }) => {
    const formattedDates = getNextDates(dates);

    const options = {
        title: {
            text: 'Injury Probability Over 30 Days',
            style: {
                fontSize: '25px'
            }
        },
        xAxis: {
            categories: formattedDates,
            title: {
                text: 'Day',
                style: {
                    fontSize: '22px'
                }
            },
            labels: {
                style: {
                    fontSize: '20px'
                }
            }
        },
        yAxis: {
            min: 0,
            max: 1,
            title: {
                text: 'Probability',
                style: {
                    fontSize: '21px'
                }
            },
            labels: {
                style: {
                    fontSize: '20px'
                }
            }
        },
        series: Object.keys(data).map((cyclistId) => ({
            name: `Cyclist ID ${cyclistId}`,
            data: data[cyclistId],
            dataLabels: {
                style: {
                    fontSize: '14px'
                }
            }
        })),
        legend: {
            itemStyle: {
                fontSize: '16px'
            }
        }
    };

    return <HighchartsReact highcharts={Highcharts} options={options} />;
};

export default ProbabilityChart;
