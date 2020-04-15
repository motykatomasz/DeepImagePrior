const highlightCode = {
  // Plugin name
  name: 'highlightCode',
  // Extend core features
  extend(api) {
    api. onContentUpdated(() => {
      document.querySelectorAll('.language-python').forEach((block) => {
        hljs.highlightBlock(block);
      });
    })
  }
}

const collapse = {
  // Plugin name
  name: 'collapse',
  // Extend core features
  extend(api) {
    api.processMarkdown(text => {
      return text
        .replace(/<collapse>/g, "<div class=collapse onclick='console.log(`hello`)'>")
        .replace(/<\/collapse>/g, "<p>click me</p></div>");
    })

    api. onContentUpdated(() => {
      document.querySelectorAll('.collapse .pre-wrapper').forEach((elem) => {
        let button = document.createElement("button");
        button.innerText = "open";
        button.className = "opener"
        let style = window.getComputedStyle(elem);
        elem.prepend(button);
        button.setAttribute('data-state', 'closed')
        button.setAttribute('data-closed-height', `${button.getBoundingClientRect().height + (+style.marginTop.replace('px','')) + 100}px`);
        button.setAttribute('data-open-height', `${elem.getBoundingClientRect().height}px`);
        elem.parentElement.style = `height:${button.getAttribute('data-closed-height')}`
        button.onclick = function() {
          if (button.getAttribute('data-state') == 'open') {
            elem.parentElement.style = `height:${button.getAttribute('data-closed-height')}`;
            button.setAttribute('data-state', 'closed')
            button.innerHTML = "open"
          } else if (button.getAttribute('data-state') == 'closed') {
            elem.parentElement.style = `height:${button.getAttribute('data-open-height')}`;
            button.setAttribute('data-state', 'open')
            button.innerHTML = "close"
          }
        }
      });
    })
  }
}

new Docute({
    target: '#docute',
    plugins: [
      docuteKatex(),
      showPlot,
      highlightCode,
      collapse,
    ],
    nav: [
      {
        title: 'Home',
        link: '/'
      },
      {
        title: 'Code',
        link: 'https://github.com/motykatomasz/DeepImagePrior'
      },
      {
        title: 'Authors',
        link: '/AUTHORS'
      }
    ],
    sidebar: [
        {
            title: 'Introduction',
            link: '/#introduction'
        },
        {
            title: 'References',
            link: '/#references'
        }
    ]
})