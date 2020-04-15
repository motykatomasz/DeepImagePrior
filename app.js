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

new Docute({
    target: '#docute',
    plugins: [
      docuteKatex(),
      showPlot,
      highlightCode,
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