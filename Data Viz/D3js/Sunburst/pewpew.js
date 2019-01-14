var nodeData = {
  'name': 'TOPICS', 'Children': [{
      'name': 'Topic A',
        'Children': [{'name': 'Sub A1', 'size':4},{'name': 'Sub A2', 'size':4}]
    }, {
      'name': 'Topic B',
            'Children': [{'name': 'Sub B1', 'size':3},{'name': 'Sub B2', 'size':3}]
    },{
      'name': 'Topic C',
            'Childrin': [{'name': 'Sub C1', 'size':4},{'name': 'Sub C2', 'size': 3}]
    }]
};

width = 500;
height = 500;
r = Math.min(width, height) / 2;
color = d3.scaleOrdinal(d3.schemeCategory20b);

var svg = d3.select('body')
    .append('svg')
    .attr('width', width)
    .attr('height', height);

var g = svg.append('g')
    .attr('transform', 'translate('+ width/2 + ','+height/2+')');