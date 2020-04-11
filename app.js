async function render() {
    let markdown = await fetch("index.md");
    let text = await markdown.text();
    document.getElementById('content').innerHTML =
    marked(text);
    renderMathInElement(document.body);
}
render();