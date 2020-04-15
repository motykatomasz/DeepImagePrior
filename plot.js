const showPlot = {
    // Plugin name
    name: 'showPlot',
    // Extend core features
    extend(api) {
      api.processMarkdown(text => {
        return text.replace(/{selector}/g, `
        <select id="photo-selector" onchange="get_data(this.value)">
          <option value="Barbara">Barbara</option>
          <option value="boat">Boat</option>
          <option value="cameraman">Cameraman</option>
          <option value="couple">Couple</option>
          <option value="fingerprint">Fingerprint</option>
          <option value="hill">Hill</option>
          <option value="house">House</option>
          <option value="Lena">Lena</option>
          <option value="man">Man</option>
          <option value="montage">Montage</option>
          <option value="peppers">Peppers</option>
        </select>
        `.trim())
        .replace(/{comparison}/g, `
        <table style="width:100%">
          <tr>
            <td></td>
            <th>Ground Truth</th>
            <th>Noisy</th>
            <th>Deep Image Prior</th>
            <th>Ours</th>
          </tr>
          <tr>
            <td></td>
            <td><img class="comparison" src="deeplearning/Barbara/GT.png"></td>
            <td><img class="comparison" src="deeplearning/Barbara/noisy.png"></td>
            <td><img class="comparison" src="deeplearning/Barbara/deep_prior.png"></td>
            <td><img class="comparison" src="deeplearning/Barbara/11000.png"></td>
          </tr>
          <tr>
            <td><strong>psnr:</strong></td>
            <td></td>
            <td></td>
            <td id="deep-image-prior-psnr"></td>
            <td id="ours-psnr"></td>
          </tr>
        </table> 
        `.trim())
        .replace(/{plot}/g, `<div id="plot"></div>`)
      })
    }
  }

function plot(data) {
    let p = d3.select("#plot");
    if (!data || !(p.node())) {
      return;
    }
    p.selectAll('*').remove();

    let margin = {top: 10, right: 30, bottom: 80, left: 60};
    let height = 400 - margin.top - margin.bottom;
    let width = p.node().clientWidth - 400 - margin.left - margin.right;

    let selector = d3.select("#photo-selector").node()

    p.selectAll().remove();
    let svg = p
    .append("svg")
      .attr("width", '100%')
      .attr("height", height + margin.top + margin.bottom)
      .on("mousemove", function() {
        let pt = svg.node().createSVGPoint();
        pt.x = d3.event.clientX;
        pt.y = d3.event.clientY;
        let rel = pt.matrixTransform(maingroup.node().getScreenCTM().inverse());
        rel.x = Math.min(Math.max(rel.x, 0), width - 1);
        highlight.attr('x1',rel.x).attr('x2',rel.x);
        let iter = Math.round(x.invert(rel.x));
        highlightPoint.attr('cx', rel.x).attr('cy', y(+data[iter].loss));
        highlightBox.attr('transform', `translate(${Math.max(rel.x-75,75)}, ${rel.y})`);
        d3.selectAll('.value-holder').data([`iteration: ${data[iter].iteration}`,`loss: ${(+data[iter].loss).toFixed(2)}`,`learning rate: ${(+data[iter].lr).toExponential(2)}`]).text(d => d);
        image.attr('href',`deeplearning/${selector.value}/${((iter/250)|0)*250}.png`)
      })

    let image = svg
      .append('image')
      .attr('x', width + margin.left + margin.right)
      .attr('y', 0)
      .attr('width', 400)
      .attr('height', 400)
      .attr('href','deeplearning/Barbara/0.png')
    let maingroup = svg.append("g")
      .attr("pointer-events","fill")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    let x = d3.scaleLinear()
      .domain([0, 11000])
      .range([ 0, width ]);

    maingroup.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("y", 0)
      .attr("x", 9)
      .attr("dy", ".35em")
      .attr("transform", "rotate(90)")
      .style("text-anchor", "start");

    let y = d3.scaleLinear()
      .domain([0, d3.max(data, function(d) { return +d.loss; })])
      .range([ height, 0 ]);

    maingroup.append("g")
      .call(d3.axisLeft(y));

      maingroup
        .append("text")
        .text("loss")
        .attr("text-anchor", "middle")
        .attr("transform",`translate(${-margin.left*3/4}, ${margin.top + height/2}) rotate(-90)`)

      maingroup
        .append("text")
        .text("iterations")
        .attr("text-anchor", "middle")
        .attr("transform",`translate(${width/2}, ${margin.top + height + margin.bottom*2/3})`)

    maingroup.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", "steelblue")
      .attr("stroke-width", 1.5)
      .attr("d", d3.line()
        .x(function(d) { return x(+d.iteration) })
        .y(function(d) { return y(+d.loss) })
        )

    let highlight = maingroup
        .append("line")
        .attr('y1',0)
        .attr('y2',height)
        .attr('x1',0)
        .attr('x2',0)
        .attr('stroke','red');

    let highlightPoint = maingroup
        .append("circle")
        .attr('cx',0)
        .attr('cy',0)
        .attr('r',3)
        .attr('fill','red')
        .attr('stroke','none');

    let highlightBox = svg.append('g').attr('transform', 'scale(0)');

    highlightBox
        .append('rect')
        .attr('x',0)
        .attr('y',0)
        .attr('fill', 'gray')
        .attr('opacity',0.25)
        .attr('rx', 15)
        .attr('width',130)
        .attr('height',50);

    highlightBox
        .append('text')
        .attr('font-size',11)
        .attr('y',0)
        .attr('text-anchor', 'middle')
        .attr('fill','black')
        .selectAll('tspan')
        .data(['a', 'b', 'c'])
        .enter()
        .append('tspan')
        .attr('class','value-holder')
        .attr('x',65)
        .attr('dy', '1.2em')
        .text(d => d);

}

let data = null;

async function get_data(value) {
  data = await d3.csv(`deeplearning/${value}/data.csv`);
  d3.selectAll(".comparison").data(["GT","noisy","deep_prior","11000"]).attr("src",d =>`deeplearning/${value}/${d}.png`)
  plot(data);
  psnr = await d3.csv(`deeplearning/${value}/psnr.csv`);
  psnr.forEach(d => d3.select(`#${d.method}-psnr`).text((+d.psnr).toFixed(2)));
}
get_data("Barbara")
window.addEventListener('resize',() => plot(data));