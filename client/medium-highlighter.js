const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))

const highlightColor = "rgb(213, 234, 255)";

const template = `
  <template id="highlightTemplate">
    <span class="highlight" style="display: inline"></span>
  </template>
  <link
    rel="stylesheet"
    href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
    integrity="sha384-pzjw8f+ua7Kw1TIq6vI6gztFkaF6M9bpkFVQoNGX7Ev0bbtvU2mgjtP3fI8bhT"
    crossorigin="anonymous"
  />
  <button id="mediumHighlighter" type="button" class="btn btn-secondary" data-bs-toggle="tooltip" data-bs-placement="top" data-bs-custom-class="custom-tooltip">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-lg" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M4.475 5.458c-.284 0-.514-.237-.47-.517C4.28 3.24 5.576 2 7.825 2c2.25 0 3.767 1.36 3.767 3.215 0 1.344-.665 2.288-1.79 2.973-1.1.659-1.414 1.118-1.414 2.01v.03a.5.5 0 0 1-.5.5h-.77a.5.5 0 0 1-.5-.495l-.003-.2c-.043-1.221.477-2.001 1.645-2.712 1.03-.632 1.397-1.135 1.397-2.028 0-.979-.758-1.698-1.926-1.698-1.009 0-1.71.529-1.938 1.402-.066.254-.278.461-.54.461h-.777ZM7.496 14c.622 0 1.095-.474 1.095-1.09 0-.618-.473-1.092-1.095-1.092-.606 0-1.087.474-1.087 1.091S6.89 14 7.496 14"/>
    </svg>
  </button>
`;

const styled = ({ display = "none", left = 0, top = 0 }) => `
  #mediumHighlighter {
    align-items: center;
    background-color: white;
    border-radius: 50%;
    border: none;
    cursor: pointer;
    display: ${display};
    justify-content: center;
    left: ${left}px;
    padding: 5px 10px;
    position: fixed;
    top: ${top}px;
    width: 40px;
    z-index: 9999;
  }
  .text-marker {
    fill: white;
  }
  .text-marker:hover {
    fill: rgb(213, 234, 255);
  }
`;

// const template = `
//   <template id="highlightTemplate">
//     <span class="highlight" style="background-color: ${highlightColor}; display: inline"></span>
//   </template>
//   <button id="mediumHighlighter" type="button" class="btn btn-secondary" data-bs-toggle="tooltip" data-bs-placement="top" data-bs-custom-class="custom-tooltip">
//     <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-lg" viewBox="0 0 16 16">
//       <path fill-rule="evenodd" d="M4.475 5.458c-.284 0-.514-.237-.47-.517C4.28 3.24 5.576 2 7.825 2c2.25 0 3.767 1.36 3.767 3.215 0 1.344-.665 2.288-1.79 2.973-1.1.659-1.414 1.118-1.414 2.01v.03a.5.5 0 0 1-.5.5h-.77a.5.5 0 0 1-.5-.495l-.003-.2c-.043-1.221.477-2.001 1.645-2.712 1.03-.632 1.397-1.135 1.397-2.028 0-.979-.758-1.698-1.926-1.698-1.009 0-1.71.529-1.938 1.402-.066.254-.278.461-.54.461h-.777ZM7.496 14c.622 0 1.095-.474 1.095-1.09 0-.618-.473-1.092-1.095-1.092-.606 0-1.087.474-1.087 1.091S6.89 14 7.496 14"/>
//     </svg>
//   </button>
// `;

// const styled = ({ display = "none", left = 0, top = 0 }) => `
//   #mediumHighlighter {
//     align-items: center;
//     background-color: white;
//     border-radius: 50%;
//     border: none;
//     cursor: pointer;
//     display: ${display};
//     justify-content: center;
//     left: ${left}px;
//     padding: 5px 10px;
//     position: fixed;
//     top: ${top}px;
//     width: 40px;
//     z-index: 9999;
//   }
//   .text-marker {
//     fill: white;
//   }
//   .text-marker:hover {
//     fill: ${highlightColor};
//   }
// `;

class MediumHighlighter extends HTMLElement {
  constructor() {
    super();
    this.render();
  }

  get markerPosition() {
    return JSON.parse(this.getAttribute("markerPosition") || "{}");
  }

  get styleElement() {
    return this.shadowRoot.querySelector("style");
  }

  get highlightTemplate() {
    return this.shadowRoot.getElementById("highlightTemplate");
  }

  static get observedAttributes() {
    return ["markerPosition"];
  }

  render() {
    this.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = styled({});
    this.shadowRoot.appendChild(style);
    this.shadowRoot.innerHTML += template;
    this.shadowRoot
      .getElementById("mediumHighlighter")
      .addEventListener("click", () => this.highlightSelection());
  }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === "markerPosition") {
      this.styleElement.textContent = styled(this.markerPosition);
    }
  }

  highlightSelection() {
    const userSelection = window.getSelection();
    const selectedText = userSelection.toString();
    if (selectedText) {
      this.sendToServer(selectedText, (result) => {
        const color = result === 'Positive' ? '#76e64e' : '#e6584e';
        for (let i = 0; i < userSelection.rangeCount; i++) {
          this.highlightRange(userSelection.getRangeAt(i), color);
        }
        window.getSelection().empty();
        setTimeout(() => {
          document.querySelectorAll('.highlight-temporary').forEach(elem => {
            elem.style.backgroundColor = '';
          });
        }, 2000);
      });
    }
    // const userSelection = window.getSelection();
    // const selectedText = userSelection.toString();
    // if (selectedText) {
    //   this.sendToServer(selectedText);
    // }
    // for (let i = 0; i < userSelection.rangeCount; i++) {
    //   this.highlightRange(userSelection.getRangeAt(i));
    // }
    // window.getSelection().empty();
  }

  highlightRange(range, color) {
    const clone = this.highlightTemplate.cloneNode(true).content.firstElementChild;
    clone.style.backgroundColor = color;
    clone.classList.add('highlight-temporary');
    clone.appendChild(range.extractContents());
    range.insertNode(clone);
    // const clone =
    //   this.highlightTemplate.cloneNode(true).content.firstElementChild;
    // clone.appendChild(range.extractContents());
    // range.insertNode(clone);
  }

  sendToServer(text, callback) {
    fetch(`http://127.0.0.1:8000/sentiment/${encodeURIComponent(text)}`)
      .then(response => response.json())
      .then(data => {
        console.log(`Server response: ${data.result}`);
        if (callback) callback(data.result);
      })
      .catch(error => {
        console.error('Error:', error);
      });
    // fetch(`http://127.0.0.1:8000/sentiment/${encodeURIComponent(text)}`)
    //   .then(response => response.json())
    //   .then(data => {
    //     console.log(`Server response: ${data.result}`);
    //   })
    //   .catch(error => {
    //     console.error('Error:', error);
    //   });
  }
}

window.customElements.define("medium-highlighter", MediumHighlighter);
