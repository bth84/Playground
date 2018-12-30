var dims = {
    width: 800,
    height: 600,
    color: d3.scaleQuantize().range([
        // "rgb(255,245,240)",
        // "rgb(254,224,210)",
        // "rgb(252,187,161)",
        // "rgb(252,146,114)",
        // "rgb(251,106,74)",
        // "rgb(239,59,44)",
        // "rgb(203,24,29)",
        // "rgb(165,15,21)",
        // "rgb(103,0,13)"
        "#f7fcfd",
        "#e5f5f9",
        "#ccece6",
        "#99d8c9",
        "#66c2a4",
        "#41ae76",
        "#238b45",
        "#006d2c",
        "#00441b"
    ])
}

var svg =  d3.select('#chart')
            .append('svg')
            .attr('width', dims.width)
            .attr('height', dims.height);
