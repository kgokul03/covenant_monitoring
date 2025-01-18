import React, { useState, useEffect } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Box,
} from "@mui/material";
import { Calendar, momentLocalizer , Views} from "react-big-calendar";
import moment from "moment";
import "react-big-calendar/lib/css/react-big-calendar.css";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DashboardIcon from "@mui/icons-material/Dashboard";
import BarChartIcon from "@mui/icons-material/BarChart";
import jsPDF from "jspdf";
import "jspdf-autotable";




const localizer = momentLocalizer(moment);

const App: React.FC = () => {
  const [events, setEvents] = useState([]);
  const [jsonData, setJsonData] = useState<any>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [currentDate, setCurrentDate] = useState(new Date()); // State for current date
  const [currentView, setCurrentView] = useState("month"); // State for current view
  const [selectedEvent, setSelectedEvent] = useState<any>(null);
  const [isEventModalOpen, setIsEventModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");

  const handleDownloadPDF = () => {
    setActiveTab("reports"); // Set "Reports" as the active tab
  
    const doc = new jsPDF();
  
    // Add title
    doc.text("Calendar Events Report", 10, 10);
  
    // Add events data
    events.forEach((event:any, index) => {
      const y = 20 + index * 10; // Dynamic vertical positioning
      doc.text(
        `${index + 1}. ${event.title} - ${new Date(event.start).toLocaleDateString()}`,
        10,
        y
      );
    });
  
    // Save PDF
    doc.save("calendar_events.pdf");
  };
  



  useEffect(() => {
    // Fetch calendar data initially
    fetchCalendarData();
  }, []);
    
  const fetchCalendarData = () => {
    fetch("http://127.0.0.1:5000/api/get-data")
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "success") {
          const calendarEvents = data.data.map((item: { issuer: string; date: string; comment: string;eventType: any; isComplaint: string;borrower:string}) => ({
            title: item.issuer,
            start: new Date(item.date),
            end: new Date(item.date),
            date: item.date,
            eventType: item.eventType,
            comment: item.comment,    
            isComplaint: item.isComplaint ,
            borrower: item.borrower
          }));
          setEvents(calendarEvents);
        }
      })
      .catch((error) => console.error("Error fetching data:", error));
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };
  
  const handleShowMore = (events: any[], date: Date) => {
    // Logic to show more events directly in the month view
    return (
      <div>
        {events.map((event, index) => (
          <div key={index}>
            <strong>{event.title}</strong> - {event.start.toLocaleString()}
          </div>
        ))}
      </div>
    );
  };

  const handleEventClick = (event: any) => {
    // When an event is selected from the calendar
    console.log(event);
    setSelectedEvent(event); // Set the selected event data
    setIsEventModalOpen(true); // Open the modal
  };

  const handleFileUpload = () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("pdf", selectedFile);

    fetch("http://127.0.0.1:5000/api/upload-pdf", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data.message);
        setJsonData(data.data);
        setIsModalOpen(true);
       
      })
      .catch((error) => console.error("Error uploading file:", error));
  };

  const handleConfirm = () => {
    fetch("http://127.0.0.1:5000/api/save-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jsonData),
    })
      .then((response) => response.json())
      .then((data) => {console.log("Data saved:", data);fetchCalendarData();})
      .catch((error) => console.error("Error saving data:", error));
    setIsModalOpen(false);
  };
  
  

  return (
    <div>
      {/* Top Bar */}
      <AppBar position="static">
        <Toolbar>
          <Typography
            variant="h5"
            component="div"
            style={{ flexGrow: 1, fontWeight: "bold", letterSpacing: "1px" }}
          >
            Covenant Monitoring Dashboard
          </Typography>
        </Toolbar>
      </AppBar>

      <Grid container spacing={3} style={{ padding: "20px", alignItems: "left" }}>
  {/* Sidebar */}
  <Grid item xs={2}>
    <Card>
      <CardContent>
        <Button
          fullWidth
          startIcon={<DashboardIcon />}
          variant={activeTab === "dashboard" ? "contained" : "outlined"} // Highlight active tab
          color={activeTab === "dashboard" ? "primary" : "secondary"} // Change color for active tab
          style={{ marginTop: "10px" }}
          onClick={() => setActiveTab("dashboard")} // Set active tab
        >
          Dashboard
        </Button>
        <Button
          fullWidth
          startIcon={<BarChartIcon />}
          variant={activeTab === "reports" ? "contained" : "outlined"} // Highlight active tab
          color={activeTab === "reports" ? "primary" : "secondary"} // Change color for active tab
          style={{ marginTop: "10px" }}
          onClick={handleDownloadPDF} // Trigger PDF download
        >
          Reports
        </Button>
      </CardContent>
    </Card>
  


          {/* Events Section */}
          {/* <Card style={{ marginTop: "20px" }}>
            <CardContent>
              <Typography
                color={"primary"}
                style={{  marginBottom: "5px" }}
              >
                Upcoming Events
              </Typography>
              <Typography variant="body1" style={{ color: "primary" }}>
                Event 1
              </Typography>
              <Typography variant="body1" style={{ color: "primary" }}>
                Event 2
              </Typography>
              <Typography variant="body1" style={{ color: "primary" }}>
                Event 3
              </Typography>
            </CardContent>
          </Card> */}

        {/* Events Section */}
<Card style={{ marginTop: "20px" }}>
  <CardContent>
    <Typography color="primary" style={{ marginBottom: "5px" }}>
      Upcoming Events
    </Typography>
    {events
      .filter((event:any) => new Date(event.start) > new Date()) // Filter future events
      .sort((a:any, b:any) => new Date(a.start).getTime() - new Date(b.start).getTime()) // Sort by start date
      .slice(0, 3) // Take the top 3 events
      .map((event:any, index) => (
        <Typography
          key={index}
          variant="body1"
          style={{ color: "primary", marginBottom: "5px" }}
        >
          {event.title} - {new Date(event.start).toLocaleDateString()}
        </Typography>
      ))}
    {events.filter((event:any) => new Date(event.start) > new Date()).length === 0 && (
      <Typography variant="body2" style={{ color: "#888" }}>
        No upcoming events
      </Typography>
    )}
  </CardContent>
</Card>

        </Grid>

        {/* Main Content */}
        <Grid item xs={10}>
          <Grid container spacing={3}>
            {/* File Upload Section */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography
                    variant="h6"
                    style={{ marginBottom: "10px" }}
                  >
                    Upload PDF
                  </Typography>
                  <Box display="flex" alignItems="center" mt={2}>
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={handleFileChange}
                      style={{
                        marginRight: "10px",
                        fontSize: "16px",
                        
                      }}
                    />
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<CloudUploadIcon />}
                      onClick={handleFileUpload}
                      disabled={!selectedFile}
                     
                    >
                      Upload
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Calendar */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography
                    variant="h6"
                    style={{  marginBottom: "10px" }}
                  >
                    Loan Agenda
                  </Typography>
                  <Calendar
                     localizer={localizer}
                                 events={events}
                                 date={currentDate} // Controlled date
                                 view={currentView} // Controlled view
                                 onNavigate={(date) => setCurrentDate(date)} // Update date on navigation
                                 onView={(view) => setCurrentView(view)} // Update view on change
                                 startAccessor="start"
                                 endAccessor="end"
                                 views={["month", "week", "day"]} // Enable view switching
                                 onShowMore={handleShowMore}
                                 onSelectEvent={handleEventClick} 
                    style={{ height: "500px", margin: "20px 0",  }}
                  />
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Modal for Data */}
      <Dialog open={isModalOpen} onClose={() => setIsModalOpen(false)}  maxWidth="lg" fullWidth>
        <DialogTitle style={{ fontWeight: "bold" }}>Event Details </DialogTitle>
        <DialogContent>
        {jsonData && (
        <TableContainer component={Paper} style={{ marginBottom: "20px" }}>
          <Table>
            <TableHead>
              <TableRow>
                {["Issuer", "Syndicate Agent", "Date", "Event Type", "Comment", "Is Complaint"].map((heading) => (
                  <TableCell key={heading} style={{ fontWeight: "bold", backgroundColor: "#f5f5f5" }}>
                    {heading}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow>
                {["issuer","borrower", "date", "eventType", "comment", "isComplaint"].map((key) => (
                  <TableCell key={key}>
                    <input
                      type="text"
                      value={jsonData[key] || ""}
                      onChange={(e) => setJsonData({ ...jsonData, [key]: e.target.value })}
                      style={{
                        width: "100%",
                        padding: "5px",
                        border: "1px solid #ccc",
                        borderRadius: "4px",
                        boxSizing: "border-box",
                      }}
                    />
                  </TableCell>
                ))}
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>)}
        </DialogContent>
        <DialogActions>
          <Button
            variant="contained"
            color="primary"
            onClick={handleConfirm}
            style={{ fontWeight: "bold" }}
          >
            Confirm
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={() => setIsModalOpen(false)}
            style={{ fontWeight: "bold" }}
          >
            Cancel
          </Button>
        </DialogActions>
      </Dialog>

      {/* Event Details Modal */}
      <Dialog open={isEventModalOpen} onClose={() => setIsEventModalOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle style={{ fontWeight: "bold" }}>Event Details</DialogTitle>
        <DialogContent style={{ maxHeight: "100vh", overflowY: "auto" }}>
          {selectedEvent && (
            <TableContainer component={Paper} style={{ marginBottom: "20px" , maxHeight: "100vh", overflowY: "auto", textAlign: "center"}}>
              <Table>
                <TableHead>
                  <TableRow>
                    {["Issuer","Syndicate Agent", "Date", "Event Type", "Comment", "Is Complaint"].map((heading) => (
                      <TableCell key={heading} style={{ fontWeight: "bold", backgroundColor: "#f5f5f5" }}>
                        {heading}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    
                    {/* Editable fields */}
                    {["title","borrower", "date", "eventType", "comment", "isComplaint"].map((key) => (
                      <TableCell key={key}>
                        <input
                          readOnly
                          type="text"
                          value={selectedEvent[key] || ""}
                          onChange={(e) =>
                            setSelectedEvent({ ...selectedEvent, [key]: e.target.value })
                          }
                          style={{
                            width: "100%",
                            padding: "5px",
                            border: "1px solid #ccc",
                            borderRadius: "4px",
                          }}
                        />
                      </TableCell>
                    ))}
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </DialogContent>
        <DialogActions>
          
          <Button
            variant="contained"
            color="secondary"
            onClick={() => setIsEventModalOpen(false)}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>
      {/* Footer */}
      <Box
        component="footer"
        style={{
          marginTop: "auto",
          backgroundColor: "#f1f1f1",
          padding: "10px",
          textAlign: "center",
        }}
      >
        <Typography variant="body2" style={{ color: "#888" }}>
          © 2025 Covenant Monitoring Dashboard. All Rights Reserved.
        </Typography>
</Box>
    </div>
  );
};

export default App;
