function accept_tweet()
{
    var tweet = document.getElementById("tweet").value;
    
        url = 'http://127.0.0.1:8000/' + encodeURIComponent(tweet);

    document.location.href = url;
}