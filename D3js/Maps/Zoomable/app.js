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
    });
});