new Docute({
    target: '#docute',
    plugins: [
      docuteKatex()
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