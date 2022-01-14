# hexo-generator-zipsearch

> Ported Plugin of hexo-generator-search-zip, Thank SuperKieran for designing such a flawless plugin.
> I modified part of the content from the hexo-generator-search-zip.

[![npm](https://img.shields.io/npm/v/hexo-generator-zip-search.svg)](https://www.npmjs.com/package/hexo-generator-zip-search)
[![npm](https://img.shields.io/npm/dm/hexo-generator-zip-search.svg)](https://www.npmjs.com/package/hexo-generator-zip-search)

Generate zip search data for Hexo 3.0. This plugin is used for generating a search index file, which contains all the neccessary data of your articles that you can use to write a local search engine for your blog.

- [Demo](https://v-vincen.life/) - try out the search engine in this site's navigation bar.
- [Demo Zip output](https://github.com/V-Vincen/hexo-generator-zip-search/blob/master/output/search.flv)

## Guide
https://v-vincen.life/

## Installation
``` bash
$ npm install hexo-generator-zip-search --save
```

## Options
You can configure this plugin in your root `_config.yml`.
``` yaml
search:
  path: search.json
  zipPath: search.flv
  versionPath: searchVersion.json
  field: post
  trigger: auto   # if 'auto', trigger search by changing input; if 'manual', trigger search by pressing enter key or search button
  top_n_per_article: 1  # show top n results per article
```

- **field** - the search scope you want to search, you can chose:
  * **post** (Default) - will only covers all the posts of your blog.
  * **page** - will only covers all the pages of your blog.
  * **all** - will covers all the posts and pages of your blog.

## FAQ

### What's this plugin supposed to do? 
This plugin is used for generating a zip file from your Hexo blog that provides data for searching.

### Where's this file saved to?
After executing `hexo g` you will get the generated result at your public folder.

