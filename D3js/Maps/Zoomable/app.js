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
};

var projection = d3.geoAlbersUsa()
    .translate([0,0]);
var path=d3.geoPath(projection);
var r_scale=d3.scaleLinear()
    .domain([0,8000000])
    .range([5,20]);

var svg =  d3.select('#chart')
            .append('svg')
            .attr('width', dims.width)
            .attr('height', dims.height);

var zoom_map=d3.zoom()
    .scaleExtent([0.3,6])
    .translateExtent([[-1000,-1000],[1000,1000]])
    .on("zoom",function(){
    // console.log(d3.event);
    var offset=[
        d3.event.transform.x,
        d3.event.transform.y
    ];
    var scale=d3.event.transform.k*2000;

    // offset[0]+=d3.event.dx;
    // offset[1]+=d3.event.dy;

    projection.translate(offset)
        .scale(scale);

    svg.selectAll("path")
        .attr("d",path);
    svg.selectAll("circle")
        .attr("cx",function(d){
            return projection([d.lon,d.lat])[0];
        })
        .attr("cy",function(d){
            return projection([d.lon,d.lat])[1];
        })
});

var map = svg.append('g')
    .attr('id', 'map')
    .attr('cursor', 'pointer')
    .call(zoom_map)
    .call(zoom_map.transform,
        d3.zoomIdentity
            .translate(dims.width/2,dims.width/2)
            .scale(1)
    );

map.append('rect')
    .attr('x',0)
    .attr('y',0)
    .attr('width', dims.width)
    .attr('height', dims.height)
    .attr('opacity', 0);

d3.json('zombie-attacks.json').then(function(zombie_data,error){
    dims.color.domain([
        d3.min(zombie_data, function(d){
            return d.num;
        }),
        d3.max(zombie_data, function(d){
            return d.num;
        })
    ]);

    d3.json('us.json').then(function(us_data, error){
       us_data.features.forEach(function(us_e, us_i){
           zombie_data.forEach(function(z_e, z_i){
               if(us_e.properties.name !== z_e.state){
                   return null;
               }
               us_data.features[us_i].properties.num=parseFloat(z_e.num);
           })
       })

        console.log(us_data);

       map.selectAll('path')
           .data(us_data.features)
           .enter()
           .append('path')
           .attr('d', path)
           .attr('fill', function(d){
               var num = d.properties.num;
               return num ? dims.color(num) : '#fff';
           })
           .attr('stroke', '#fff')
           .attr('stroke-width', 2);

       draw_cities();
    });
});

function draw_cities(){
    d3.json("us-cities.json").then(function (city_data) {
        map.selectAll("circle")
            .data(city_data)
            .enter()
            .append("circle")
            .style("fill", "#9d1c3c")
            .style("opacity", 0.8)
            .attr("cx",function(d){
                return projection([d.lon,d.lat])[0];
            })
            .attr("cy",function(d){
                return projection([d.lon,d.lat])[1];
            })
            .attr("r",function(d){
                return r_scale(d.population);
            })
            .append("title")
            .text(function(d){
                return d.city;
            });

    });
}

d3.selectAll("#buttons button.panning")
    .on("click",function(){
        var distance =100;
        var direction=d3.select(this).attr("class");
        var x=0;
        var y=0;

        if (direction == "panning up"){
            y +=distance;
        }else if (direction == "panning down"){
            y-=distance;
        }else if (direction == "panning left"){
            x +=distance;
        }else if (direction == "panning right"){
            x -=distance;
        }
        map.transition()
            .call(zoom_map.translateBy,x,y);
    });

d3.selectAll("#buttons button.zooming")
    .on("click",function(){
        var scale =1;
        var direction=d3.select(this).attr("class");


        if (direction == "zooming in"){
            scale=1.25;
        }else if (direction == "zooming out") {
            scale = 0.75;
        }map.transition()
            .call(zoom_map.scaleBy,scale);
    });
