const url = "http://127.0.0.1:8000/orders"

async function getData() {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Response status: ${response.status}`);
    }

    const json = await response.json();
    displayTotalAverageTime(json['total_avg_processing_time'], json['avg_process_time']);
    drawLineChart(json['orders']);
    drawPieChart(json['priorities'], "#prPieChart", "Orders by Priority");
    drawPieChart(json['orders_by_package_type'], "#pkPieChart", "Orders by Package Type");
  } catch (error) {
    console.error(error.message);
  }
}

function displayTotalAverageTime(total_time, two_weeks_time) {
  const totalAverageTimeElement = document.getElementById("numberDisplayTotal");
  const twoWeeksAverageTimeElement = document.getElementById("numberDisplayTwoWeeks");

  let total_days = +total_time.toFixed(2);
  let two_weeks_days = +two_weeks_time.toFixed(2);

  if (total_days <= 2) {
    totalAverageTimeElement.style.color = "#2ecc71";
  }
  else {
    totalAverageTimeElement.style.color = "#e74c3c"
  }

  if (two_weeks_days <= 2) {
    twoWeeksAverageTimeElement.style.color = "#2ecc71"
  }
  else {
    twoWeeksAverageTimeElement.style.color = "#e74c3c"
  }

  totalAverageTimeElement.innerHTML = JSON.stringify(total_days, null, 2);
  twoWeeksAverageTimeElement.innerHTML = JSON.stringify(two_weeks_days, null, 2);

}

function drawLineChart(rawData) {
  const parseDate = d3.timeParse("%Y-%m-%d %H:%M:%S");
  const formatDate = d3.timeFormat("%Y-%m-%d");

  const data = rawData.map(d => ({
    date: parseDate(d[0]),
    processingTime: +d[1]
    }));

  const groupedData = d3.group(data, d => formatDate(d.date));
  const averagedData = Array.from(groupedData, ([key, values]) => ({
    date: parseDate(key + " 00:00:00"),
    processingTime: d3.mean(values, d => d.processingTime)
    }));

  const margin = {top: 20, right: 30, bottom: 50, left: 60},
  width = 700 - margin.left - margin.right,
  height = 470 - margin.top - margin.bottom;

  const svg = d3.select("#chart")
          .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
          .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleTime()
          .domain(d3.extent(averagedData, d => d.date))
          .range([ 0, width ]);
  svg.append("g")
          .attr("transform", `translate(0,${height})`)
          .call(d3.axisBottom(x).tickFormat(d3.timeFormat("%d-%m-%Y")).ticks(5))
          .selectAll("text")
            .attr("transform", "rotate(-45)")
            .style("text-anchor", "end");

  const y = d3.scaleLinear()
          .domain([0, d3.max(averagedData, d => d.processingTime)])
          .range([ height, 0 ]);
  svg.append("g")
          .call(d3.axisLeft(y));

  svg.append('g')
          .selectAll("dot")
          .data(averagedData)
          .enter()
          .append("circle")
            .attr("cx", d => x(d.date))
            .attr("cy", d => y(d.processingTime))
            .attr("r", 5)
            .attr("fill", "steelblue");

  svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 0 - margin.left)
            .attr("x",0 - (height / 2))
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text("Processing Time (days)");

  svg.append("text")
            .attr("x", (width / 2))             
            .attr("y", 0 - (margin.top + 20  / 2))
            .attr("text-anchor", "middle")  
            .style("font-size", "20px") 
            .style("text-decoration", "underline")  
            .text("Average Processing Time by Date");
}

function drawPieChart(data, selector, title) {

var pieData = Object.entries(data[0]).map(([key, value]) => ({
    priority: key,
    count: value
}));

var width = 500,
    height = 470,
    margin = 40;

var radius = Math.min(width, height) / 2 - margin;

var svg = d3.select(selector)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

var color = d3.scaleOrdinal()
    .domain(pieData.map(d => d.priority))
    .range(d3.schemeCategory10);

var pie = d3.pie()
    .value(function(d) { return d.count; });

var data_ready = pie(pieData);

var arc = d3.arc()
    .innerRadius(0)
    .outerRadius(radius);

var label = d3.arc()
    .outerRadius(radius)
    .innerRadius(radius - 80);

svg.selectAll('text')
    .data(data_ready)
    .enter()
    .append('text')
    .attr("transform", function(d) { return "translate(" + label.centroid(d) + ")"; })
    .attr("dy", "5px")
    .attr("text-anchor", "middle")
    .text(function(d) { return d.data.priority; })
    .style("font-size", "12px");

svg
    .selectAll('whatever')
    .data(data_ready)
    .enter()
    .append('path')
    .attr('d', arc)
    .attr('fill', function(d) { return(color(d.data.priority)); })
    .attr("stroke", "white")
    .style("stroke-width", "2px")
    .style("opacity", 0.7);

svg.append("text")
    .attr("x", 0)
    .attr("y", 0 - (height / 2 - 20))
    .attr("text-anchor", "middle")
    .style("font-size", "16px")
    .style("font-weight", "bold")
    .text(title);
}

document.getElementById('order-form').addEventListener('submit', function(event) {
  event.preventDefault(); 

  const priority = document.getElementById('priority').value;

  const data = {
      priority: priority
  };

  fetch(url, {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data => {
      let displayText = "Order processing time: " + data['processing_time'].toFixed(1).toString() + " days";
      alert(displayText);
  })
  .catch((error) => {
      console.error('Error:', error);
      alert('An error occurred while submitting the order.');
  });
});

getData()