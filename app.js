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