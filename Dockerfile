FROM jekyll/jekyll:3.8

WORKDIR /srv/jekyll/site
COPY site/Gemfile .
COPY site/Gemfile.lock .
RUN chmod -R 777 /srv/jekyll/site; \
    bundle install
COPY site /srv/jekyll/site
CMD jekyll serve --livereload
