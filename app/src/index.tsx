import ReactDOM from "react-dom/client";
import ChatComponent from "../src/Components/Chat";
import "./index.css";

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);

root.render(
  <div className="flex-container bar">
    <div>
      <div>
        <h1 className="element1">EMOTIONAI</h1>
        <img className="element2" src={require('./images/transbrain-removebg-preview_1_1.png')}></img>
        {/* <img src={require('./images/title.png')}></img> */}
      </div>
      <span>
        <img className="element3" src={require('./images/marshy.png')}></img>
        <img className="element4" src={require('./images/campfire.png')}></img>
      </span>

    </div>
    <div className="chat">
      <ChatComponent />
    </div>
  </div>
);
