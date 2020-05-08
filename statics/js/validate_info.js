var attempt = 3; // Variable to count number of attempts.
// Below function Executes on click of login button.

function validate_info()
{
    var username = document.getElementById("user").value;
    var password = document.getElementById("pass").value;
    if ( username == "admin" && password == "admin")
    {
        alert ("Start Detection");
        window.location = 'home/'  //{% static 'http://127.0.0.1:8000/pract/home' %} // Redirecting to other page.
    return false;
    }
}
    else
    {
        attempt --;// Decrementing by one.
        alert("You have left "+attempt+" attempt;");
        // Disabling fields after 3 attempts.
        if( attempt == 0)
        {
            document.getElementById("username").disabled = true;
            document.getElementById("password").disabled = true;
            document.getElementById("submit").disabled = true;
            return false;
        }
    }
}