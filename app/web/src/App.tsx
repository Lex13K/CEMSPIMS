import { NavLink, Outlet } from "react-router-dom";
import { GraphProvider } from "./context/GraphStore";
import GraphPage from "./pages/GraphPage";
import ComparePage from "./pages/ComparePage";
import ConfigPage from "./pages/ConfigPage";
import JobsPage from "./pages/JobsPage";

function Shell() {
  return (
    <GraphProvider>
      <div className="layout">
        <nav className="nav">
          <h1>CEMSPIMS</h1>
          <NavLink to="/graph">Graph</NavLink>
          <NavLink to="/compare">Compare</NavLink>
          <NavLink to="/jobs">Jobs</NavLink>
          <NavLink to="/config">Config</NavLink>
        </nav>
        <div className="main">
          <Outlet />
        </div>
      </div>
    </GraphProvider>
  );
}

const App = Object.assign(Shell, {
  Graph: GraphPage,
  Compare: ComparePage,
  Config: ConfigPage,
  Jobs: JobsPage,
});

export default App;
