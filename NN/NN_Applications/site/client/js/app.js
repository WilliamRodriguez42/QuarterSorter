var ctx = document.getElementById('myChart').getContext('2d');
var myChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [{
            label: 'Number of Quarters',
            data: [],
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
        }]
    },
    options: {
        legend: {
            labels: {
                fontColor: 'white'
            }
        },
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero: true,
                    fontColor: 'white'
                },
                gridLines: {
                    color: '#888'
                }
            }],
            xAxes: [{
                ticks: {
                    fontColor: 'white'
                },
                gridLines: {
                    color: '#888'
                }
            }]
        }
    }
});

function get_state_info() {
    $.ajax({
        type: 'GET',
        url: '/state_info',
        success: function(state_info) {
            state_info = JSON.parse(state_info); 

            labels = []
            data = []
            for (state in state_info) {
                labels.push(state);
                data.push(state_info[state]);
            }

            myChart.data.labels = labels;
            myChart.data.datasets[0].data = data;
            myChart.update()
        }
    });
}

function move_quarter_up_one_step() {
    $.ajax({
        type: 'POST',
        url: '/move_quarter_up_one_step'
    });
}

function move_quarter_down_one_step() {
    $.ajax({
        type: 'POST',
        url: '/move_quarter_down_one_step'
    });
}

get_state_info(); // Immediately get the state info on startup
get_state_info_interval_id = setInterval(get_state_info, 1000); // Get state info every second