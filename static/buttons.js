const ParametersRefreshTime = 1000;

async function SendJSON(endpoint, data, method) {
    fetch(
        endpoint, {
        method: method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
};

function SpawnWalkers(e) {
    if (e.key == "Enter") {
        let data = { data: this.value };
        SendJSON("/api/spawnwalkers", data, "POST");
        this.blur();
    }

}

function SpawnVehicles(e) {
    if (e.key == "Enter") {
        let data = { data: this.value };
        SendJSON("/api/spawnvehicles", data, "POST");
        this.blur();
    }

}




function GetParameters(endpoint) {
    try {
        var HTTPreq = new XMLHttpRequest();
        HTTPreq.open("GET", endpoint, false);
        HTTPreq.send(null);
        return HTTPreq.responseText;
    }
    catch (error) {
        console.log(error);
        GetParameters(endpoint);
    }
}


function ChangeCamera() {

    if (document.querySelector(".tab_selected").id != this.id) {
        var e = (this.id == "agent") ? 0 : (this.id == "spectator") ? 1 : 2;
        SendJSON("/api/changecam", { data: e }, "POST");

        const tabs = document.querySelectorAll(".tab");
        tabs.forEach(tab => tab.className = "tab");

        this.className += " tab_selected";
        this.blur();
    }
}


function ChangeDirection() {

    let selected = this.id;
    let data = { data: selected };
    SendJSON("/api/changedirection", data, "POST");

    document.querySelectorAll(".arrow").
        forEach(elem => elem.className = elem.className.split(" ").slice(0, 2).join(' '));

    this.className += " arrow-" + this.id + "-selected";


}


function ToggleFullScreenStream() {
    if (this.fullscreenElement)
        this.exitFullscreen();
    else
        this.requestFullscreen().catch(console.log);
}


document.getElementById("spawn-walkers").addEventListener("keypress", SpawnWalkers);
document.getElementById("spawn-vehicles").addEventListener("keypress", SpawnVehicles);
document.getElementById("left").addEventListener("click", ChangeDirection);
document.getElementById("up").addEventListener("click", ChangeDirection);
document.getElementById("right").addEventListener("click", ChangeDirection);
document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", ChangeCamera));
document.querySelector("#stream").addEventListener("dblclick", ToggleFullScreenStream);

function ChangeTown() {
    e = this.id;
    SendJSON("/api/changetown", { data: e }, "POST");
}

function ChangeWeather() {
    e = this.id;
    SendJSON("/api/changeweather", { data: e }, "POST");
}

document.querySelectorAll(".town-item").
    forEach(e => e.addEventListener("click", ChangeTown));

document.querySelectorAll(".weather-preset").
    forEach(e => e.addEventListener("click", ChangeWeather));


async function setProgress(name, progress, spinner_id, circle_id) {
    var current_val_str = circle_id.innerHTML;
    var current_val = parseInt(current_val_str.substring(name.length, current_val_str.length - 1));
    if (!current_val)
        current_val = 0;
    var increment = (current_val > progress) ? -1 : (current_val < progress) ? 1 : 0;
    await ChangePbarLoop(name, current_val, progress, increment, spinner_id, circle_id);

};


async function ChangePbarLoop(name, current_val, progress, increment, spinner_id, circle_id) {

    var diff = Math.abs(progress - current_val);
    if (diff >= 20)
        var delay = 10;
    if (diff < 20)
        var delay = 20;
    if (diff <= 10)
        var delay = 40;
    if (diff <= 5)
        var delay = 100;


    setTimeout(function () {

        if (current_val < 60)
            var color = "rgb(0, 255, 0)";
        if (current_val >= 60 & current_val < 90)
            var color = "rgb(255, 255, 0)";
        if (current_val >= 90)
            var color = "rgb(255, 0, 0)";

        spinner_id.style.background =
            "conic-gradient(" + color +
            current_val +
            "%,#e7eaee " +
            current_val +
            "%)";

        current_val += increment;
        circle_id.innerHTML = name + " " + current_val.toString() + "%";
        if (current_val != progress)
            ChangePbarLoop(name, current_val, progress, increment, spinner_id, circle_id);
    }, delay)
}

function SyncParameters() {
    var params = JSON.parse(GetParameters("/api/serverstats"));

    setProgress("CPU", params.cpu,
        document.querySelector("#ps_cpu"),
        document.querySelector("#cpu_meter")
    );

    setProgress("RAM", params.ram,
        document.querySelector("#ps_ram"),
        document.querySelector("#ram_meter")
    );

    setProgress("GPU", params.gpu,
        document.querySelector("#ps_gpu"),
        document.querySelector("#gpu_meter")
    );

    setProgress("VRAM", params.vram,
        document.querySelector("#ps_vram"),
        document.querySelector("#vram_meter")
    );

    document.getElementById("fps-counter").innerHTML = "FPS: " + params.fps;
    setTimeout(SyncParameters, ParametersRefreshTime);
}

SyncParameters();

BrakeAutopilot = 1;

function TogglePause() {
    this.classList.toggle('pause');
    BrakeAutopilot ^= 1;
    var data = { data: BrakeAutopilot };
    SendJSON("/api/brakeautopilot", data, "POST");
}

document.querySelector(".box")
    .addEventListener("click", TogglePause);


var autopilot=0, w=0,a=0, s=0, d=0, q=0; 
keyevents= null;

function KeyListenerLoop(e){
    
    if(e.keyCode==119){

        if(s){
            s=0;
            w=0.1;
        }
        else
        w=1;
    }

    if(e.keyCode==115){
        s=1;
        w=0;
    }
        
    if(e.keyCode==100)
        d=1;

    if(e.keyCode==97)
        a=1;
    
    if(e.keyCode==113){
        q=1;
    }

}


function SendInput(){
    setTimeout(function(){
        data = {
            w: w,
            a: a,
            s: s,
            d: d,
            q: q
        };
        SendJSON("/api/manualcontrol", data, "POST");
        a=0;
        d=0;
        q=0;
        if(autopilot)
            SendInput();
    }, 100);
}



function StartInputListener(){
    

    if(autopilot){
        window.addEventListener("keypress", KeyListenerLoop);
    }


}


function ToggleAutoPilot(){
    autopilot^=1;
    SendJSON("/api/changepilotmode", {data: !autopilot}, "POST");    
    if(!autopilot){

        window.removeEventListener("keypress", KeyListenerLoop);
        document.querySelector(".toggle-autopilot").innerHTML="Autopilot";
        document.querySelector(".ai-controller").style.display ="flex";
    }
    else{
        document.querySelector(".toggle-autopilot").innerHTML="Manual";
        document.querySelector(".ai-controller").style.display = "none";
        StartInputListener();
        w=0;
        a=0;
        s=0;
        d=0;
        q=0;
        SendInput();
    }

}

document.querySelector("#toggle-autopilot")
    .addEventListener("click", ToggleAutoPilot);


