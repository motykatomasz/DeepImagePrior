

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
  }
}

const loader = {
  name: "loader",
  extend(api) {
    api.processHTML((text) => text + `<img id="ciao" src="" onerror="load()">`)
  }
}

function load() {
  plot(data)

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
  })

  document.querySelectorAll('.language-python').forEach((block) => {
    hljs.highlightBlock(block);
  })
}

new Docute({
    target: '#docute',
    plugins: [
      docuteKatex(),
      showPlot,
      collapse,
      loader,
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
          title: 'Problems tackled by the paper',
          link: '/#problems-tackled-by-the-paper'
        },
        {
          title: 'What is Deep Image Prior',
          link: '/#what-is-deep-image-prior'
        },
        {
          title: 'How does it work?',
          link: '/#how-does-it-work'
        },
        {
          title: 'Developing the Network from the paper',
          children: [
            {
              title: 'Structure of The Network',
              link: '/#structure-of-the-network'
            },
            {
              title: 'Peculiarities From The Network Structure',
              link: '/#peculiarities-from-the-network-structure'
            },
            {
              title: 'Resolving The Peculiarities',
              link: '/#resolving-the-peculiarities'
            },
            {
              title: 'Putting It All Together',
              link: '/#putting-it-all-together'
            },
          ]
        },
        {
          title: 'Learning process',
          link: '/#learning-process'
        },
        {
          title: 'Experimental Results',
          link: '/#experimental-results'
        },
        {
            title: 'References',
            link: '/#references'
        }
    ]
})