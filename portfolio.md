---
layout: page
REMOVEME_title: portfolio # Change var name to "title" to include in navigation again
REMOVEME_permalink: /portfolio/ # Change var name to "permalink" to fix
---

{% for project in site.posts %}
{% assign catlist = project.categories | join: '&ensp;•&ensp;'%}

{% if project.redirect %}
<div class="project">
    <div class="thumbnail">
        <a href="{{ project.redirect }}" target="_blank">
        {% if project.img %}
        <img class="thumbnail" src="{{ project.img }}"/>
        {% else %}
        <div class="thumbnail blankbox"></div>
        {% endif %}    
        <span>
            <h1>{{ project.title }}</h1>
            <br/>
            <p>{{ catlist }}</p>
        </span>
        </a>
    </div>
</div>
{% else %}

<div class="project ">
    <div class="thumbnail">
        <a href="{{ site.baseurl }}{{ project.url }}">
        {% if project.img %}
        <img class="thumbnail" src="{{ project.img }}"/>
        {% else %}
        <div class="thumbnail blankbox"></div>
        {% endif %}    
        <span>
            <h1>{{ project.title }}</h1>
            <br/>
            <p>{{ catlist }}</p>
        </span>
        </a>
    </div>
</div>

{% endif %}

{% endfor %}
