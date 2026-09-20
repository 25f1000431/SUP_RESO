import "./HeroBackground.css";
import earth from "../assets/earth.png";
import satellite from "../assets/satellite.png";

export default function HeroBackground() {
  return (
   <div className="hero-bg">

  <div className="stars"></div>

  <div className="shooting shooting1"></div>
  <div className="shooting shooting2"></div>

  <div className="earth-wrapper">

    <img src={earth} className="earth" />

    <div className="orbit orbit1">
  <div className="satellite-holder front-sat">
    <img src={satellite} className="satellite" />
  </div>
</div>

<div className="orbit orbit2">
  <div className="satellite-holder back-sat">
    <img src={satellite} className="satellite" />
  </div>
</div>

<div className="orbit orbit3">
  <div className="satellite-holder top-sat">
    <img src={satellite} className="satellite" />
  </div>
</div>
<div className="node node1"></div>
<div className="node node2"></div>
      <div className="node node3"></div>
    </div>

  </div>


  );
}