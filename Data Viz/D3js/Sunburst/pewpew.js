var nodeData = {
        "name": "TOPICS", "children": [{
            "name": "Topic A",
            "children": [{"name": "Sub A1", "size": 4}, {"name": "Sub A2", "size": 4}]
        }, {
            "name": "Topic B",
            "children": [{"name": "Sub B1", "size": 3}, {"name": "Sub B2", "size": 3}, {
                "name": "Sub B3", "size": 3}]
        }, {
            "name": "Topic C",
            "children": [{"name": "Sub A1", "size": 4}, {"name": "Sub A2", "size": 4}]
        }]
    };

width = 500;
height = 500;
r = Math.min(width, height) / 2;
color = d3.scaleOrdinal(d3.schemeCategory20b);

// Create primary <g> element
var g = d3.select('svg')
    .attr('width', width)
    .attr('height', height)
    .append('g')
    .attr('transform', 'translate(' + width / 2 + ',' + height / 2 + ')');


var partition = d3.partition().size([2*Math.PI, r]);

var root = d3.hierarchy(nodeData).sum(function(d){ return d.size});

root = partition(root);
var arc = d3.arc()
    .startAngle(function(d){ return d.x0 })
    .endAngle(function(d){ return d.x1 })
    .innerRadius(function(d){ return d.y0 })
    .outerRadius(function(d){ return d.y1 });

g.selectAll('path')
    .data(root.descendants())
    .enter().append('path')
    .attr('display', function(d){ return d.depth ? null : "none";})
    .attr("d", arc)
    .style('stroke', '#fff')
    .style('fill', function(d){ return color((d.children ? d : d.parent).data.name); });
