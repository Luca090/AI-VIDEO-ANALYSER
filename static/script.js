const form =
    document.querySelector("form");

const button =
    document.getElementById(
        "analyzeButton"
    );


if (form) {

    form.addEventListener(
        "submit",
        function () {

            button.innerText =
                "Analyzing Video...";

            button.disabled = true;

        }
    );

}