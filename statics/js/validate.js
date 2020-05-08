var attempt = 3; // Variable to count number of attempts.
// Below function Executes on click of login button.


function validate_activity()
{
    var username = document.getElementById("user").value;
    var password = document.getElementById("pass").value;
    if ( username == "admin" && password == "admin")
    {
        alert ("Login successfully");
        window.location = 'home/'
    return false;
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
function data()
{
        window.location = 'home/data/'
}

function result()
{
        window.location = 'result/'
}

function graphs()
{
        window.location = 'graph/'
}
