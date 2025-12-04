import React, { useState, useEffect } from 'react';
import { Upload, CheckCircle, XCircle, Users, Award, BarChart3, FileText, LogIn, UserPlus, Menu, X } from 'lucide-react';

// Mock API calls - Replace with actual Spring Boot endpoints
const API_BASE = '/api';

const App = () => {
  const [currentUser, setCurrentUser] = useState(null);
  const [activeView, setActiveView] = useState('login');
  const [ideas, setIdeas] = useState([]);
  const [selectedIdeas, setSelectedIdeas] = useState([]);
  const [allocations, setAllocations] = useState([]);
  const [preferences, setPreferences] = useState([]);
  const [topIdeas, setTopIdeas] = useState([]);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Load data based on user role
  useEffect(() => {
    if (currentUser) {
      loadIdeas();
      loadSelectedIdeas();
      loadAllocations();
      loadPreferences();
      loadTopIdeas();
    }
  }, [currentUser]);

  const loadIdeas = async () => {
    // Mock data - replace with fetch
    setIdeas([
      { id: 1, title: 'AI-Powered Code Review', description: 'Automated code review using ML', author: 'John Doe', category: 'AI/ML', userSet: 'SET1', status: 'PENDING' },
      { id: 2, title: 'Microservices Architecture', description: 'Migrate to microservices', author: 'Jane Smith', category: 'Architecture', userSet: 'SET2', status: 'PENDING' }
    ]);
  };

  const loadSelectedIdeas = async () => {
    setSelectedIdeas([]);
  };

  const loadAllocations = async () => {
    setAllocations([]);
  };

  const loadPreferences = async () => {
    setPreferences([]);
  };

  const loadTopIdeas = async () => {
    setTopIdeas([]);
  };

  const LoginView = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [isRegistering, setIsRegistering] = useState(false);
    const [newUser, setNewUser] = useState({ username: '', password: '', role: 'NORMAL', userSet: 'SET1' });

    const handleLogin = () => {
      // Mock login - replace with actual API call
      const mockUser = { username, role: username.includes('admin') ? 'ADMIN' : username.includes('approver') ? 'APPROVER' : 'NORMAL', userSet: 'SET1' };
      setCurrentUser(mockUser);
      setActiveView('dashboard');
    };

    const handleRegister = () => {
      // Mock registration
      alert('User registered successfully! Please login.');
      setIsRegistering(false);
    };

    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
          <div className="flex items-center justify-center mb-6">
            <div className="w-16 h-16 bg-indigo-600 rounded-full flex items-center justify-center">
              <Award className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-center text-gray-800 mb-2">Engineering Change Day</h1>
          <p className="text-center text-gray-600 mb-6">Innovation Dashboard</p>

          {!isRegistering ? (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="Enter username"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="Enter password"
                />
              </div>
              <button onClick={handleLogin} className="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700 transition flex items-center justify-center gap-2">
                <LogIn className="w-5 h-5" />
                Login
              </button>
              <button onClick={() => setIsRegistering(true)} className="w-full bg-white border border-indigo-600 text-indigo-600 py-2 rounded-lg hover:bg-indigo-50 transition flex items-center justify-center gap-2">
                <UserPlus className="w-5 h-5" />
                Register New User
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <input
                  type="text"
                  value={newUser.username}
                  onChange={(e) => setNewUser({...newUser, username: e.target.value})}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                <input
                  type="password"
                  value={newUser.password}
                  onChange={(e) => setNewUser({...newUser, password: e.target.value})}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <select
                  value={newUser.role}
                  onChange={(e) => setNewUser({...newUser, role: e.target.value})}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="NORMAL">Normal User</option>
                  <option value="ADMIN">Admin</option>
                  <option value="APPROVER">Approver</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">User Set</label>
                <select
                  value={newUser.userSet}
                  onChange={(e) => setNewUser({...newUser, userSet: e.target.value})}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="SET1">Set 1</option>
                  <option value="SET2">Set 2</option>
                </select>
              </div>
              <button onClick={handleRegister} className="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700 transition">
                Register
              </button>
              <button onClick={() => setIsRegistering(false)} className="w-full bg-gray-200 text-gray-700 py-2 rounded-lg hover:bg-gray-300 transition">
                Back to Login
              </button>
            </div>
          )}

          <div className="mt-6 text-center text-sm text-gray-600">
            <p>Demo credentials:</p>
            <p>admin / admin123</p>
            <p>approver / approver123</p>
            <p>user / user123</p>
          </div>
        </div>
      </div>
    );
  };

  const SubmitIdeaView = () => {
    const [formData, setFormData] = useState({
      title: '',
      description: '',
      category: 'Technology',
      impactArea: 'Efficiency',
      estimatedBenefit: ''
    });

    const handleSubmit = async () => {
      // API call to submit idea
      alert('Idea submitted successfully!');
      setFormData({ title: '', description: '', category: 'Technology', impactArea: 'Efficiency', estimatedBenefit: '' });
      loadIdeas();
    };

    return (
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
            <Upload className="w-6 h-6 text-indigo-600" />
            Submit Your Idea
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Idea Title *</label>
              <input
                type="text"
                value={formData.title}
                onChange={(e) => setFormData({...formData, title: e.target.value})}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Description *</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({...formData, description: e.target.value})}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 h-32"
              />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Category *</label>
                <select
                  value={formData.category}
                  onChange={(e) => setFormData({...formData, category: e.target.value})}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                >
                  <option>Technology</option>
                  <option>Process</option>
                  <option>Innovation</option>
                  <option>Quality</option>
                  <option>Cost Reduction</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Impact Area *</label>
                <select
                  value={formData.impactArea}
                  onChange={(e) => setFormData({...formData, impactArea: e.target.value})}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                >
                  <option>Efficiency</option>
                  <option>Customer Satisfaction</option>
                  <option>Team Productivity</option>
                  <option>Revenue Growth</option>
                  <option>Risk Mitigation</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Estimated Benefit</label>
              <input
                type="text"
                value={formData.estimatedBenefit}
                onChange={(e) => setFormData({...formData, estimatedBenefit: e.target.value})}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                placeholder="e.g., 20% time savings, $50K annual cost reduction"
              />
            </div>
            <button onClick={handleSubmit} className="w-full bg-indigo-600 text-white py-3 rounded-lg hover:bg-indigo-700 transition font-medium">
              Submit Idea
            </button>
          </div>
        </div>
      </div>
    );
  };

  const AllIdeasView = () => {
    return (
      <div>
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <FileText className="w-6 h-6 text-indigo-600" />
            All Submitted Ideas
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {ideas.map((idea) => (
              <div key={idea.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition">
                <h3 className="font-bold text-lg text-gray-800 mb-2">{idea.title}</h3>
                <p className="text-gray-600 text-sm mb-3">{idea.description}</p>
                <div className="flex flex-wrap gap-2 mb-2">
                  <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-xs">{idea.category}</span>
                  <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs">{idea.userSet}</span>
                </div>
                <p className="text-xs text-gray-500">By: {idea.author}</p>
              </div>
            ))}
          </div>
        </div>

        {currentUser?.role === 'ADMIN' && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">Admin: Select Top 10 Ideas</h3>
            <button className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition">
              Submit Selected Ideas for Approval
            </button>
          </div>
        )}
      </div>
    );
  };

  const SubmitPreferencesView = () => {
    const [selectedPreferences, setSelectedPreferences] = useState([]);

    const togglePreference = (ideaId) => {
      if (selectedPreferences.includes(ideaId)) {
        setSelectedPreferences(selectedPreferences.filter(id => id !== ideaId));
      } else if (selectedPreferences.length < 3) {
        setSelectedPreferences([...selectedPreferences, ideaId]);
      }
    };

    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <Award className="w-6 h-6 text-indigo-600" />
          Submit Your Preferences (Max 3)
        </h2>
        <p className="text-gray-600 mb-6">Select up to 3 ideas you'd like to work on. Selected: {selectedPreferences.length}/3</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {selectedIdeas.map((idea) => (
            <div
              key={idea.id}
              onClick={() => togglePreference(idea.id)}
              className={`border-2 rounded-lg p-4 cursor-pointer transition ${
                selectedPreferences.includes(idea.id)
                  ? 'border-indigo-600 bg-indigo-50'
                  : 'border-gray-200 hover:border-indigo-300'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-bold text-gray-800">{idea.title}</h3>
                  <p className="text-sm text-gray-600 mt-1">{idea.description}</p>
                </div>
                {selectedPreferences.includes(idea.id) && (
                  <CheckCircle className="w-6 h-6 text-indigo-600 flex-shrink-0 ml-2" />
                )}
              </div>
            </div>
          ))}
        </div>
        <button
          disabled={selectedPreferences.length === 0}
          className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          Submit Preferences
        </button>
      </div>
    );
  };

  const AllocationView = () => {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <Users className="w-6 h-6 text-indigo-600" />
          Team Allocations
        </h2>
        {allocations.length > 0 ? (
          <div className="space-y-4">
            {allocations.map((allocation, idx) => (
              <div key={idx} className="border border-gray-200 rounded-lg p-4">
                <h3 className="font-bold text-lg mb-2">{allocation.ideaTitle}</h3>
                <p className="text-gray-600">Allocated Members: {allocation.members.length}</p>
                <div className="flex flex-wrap gap-2 mt-2">
                  {allocation.members.map((member, i) => (
                    <span key={i} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                      {member}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-600">No allocations yet. Waiting for admin approval.</p>
        )}

        {currentUser?.role === 'APPROVER' && (
          <div className="mt-6 flex gap-4">
            <button className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition flex items-center gap-2">
              <CheckCircle className="w-5 h-5" />
              Approve Allocations
            </button>
            <button className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition flex items-center gap-2">
              <XCircle className="w-5 h-5" />
              Request Review
            </button>
          </div>
        )}
      </div>
    );
  };

  const TopIdeasView = () => {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <Award className="w-6 h-6 text-yellow-600" />
          Top 3 Best Ideas
        </h2>
        {topIdeas.length > 0 ? (
          <div className="space-y-4">
            {topIdeas.map((idea, idx) => (
              <div key={idx} className="border-2 border-yellow-400 rounded-lg p-6 bg-gradient-to-r from-yellow-50 to-white">
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-10 h-10 bg-yellow-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                    {idx + 1}
                  </div>
                  <h3 className="font-bold text-xl text-gray-800">{idea.title}</h3>
                </div>
                <p className="text-gray-600 ml-13">{idea.description}</p>
                <p className="text-sm text-gray-500 mt-2 ml-13">Votes: {idea.votes}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-600">Top ideas will be displayed after admin voting.</p>
        )}
      </div>
    );
  };

  const DashboardView = () => {
    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-md">
          <div className="max-w-7xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-indigo-600 rounded-lg flex items-center justify-center">
                  <Award className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-800">Engineering Change Day</h1>
                  <p className="text-sm text-gray-600">Welcome, {currentUser?.username} ({currentUser?.role})</p>
                </div>
              </div>
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="lg:hidden p-2"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
              <button
                onClick={() => {
                  setCurrentUser(null);
                  setActiveView('login');
                }}
                className="hidden lg:block bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition"
              >
                Logout
              </button>
            </div>
          </div>
        </header>

        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Sidebar */}
            <aside className={`${mobileMenuOpen ? 'block' : 'hidden'} lg:block lg:w-64 flex-shrink-0`}>
              <div className="bg-white rounded-xl shadow-lg p-4 space-y-2">
                <button
                  onClick={() => {
                    setActiveView('submit');
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition flex items-center gap-2 ${
                    activeView === 'submit' ? 'bg-indigo-600 text-white' : 'hover:bg-gray-100'
                  }`}
                >
                  <Upload className="w-5 h-5" />
                  Submit Idea
                </button>
                <button
                  onClick={() => {
                    setActiveView('ideas');
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition flex items-center gap-2 ${
                    activeView === 'ideas' ? 'bg-indigo-600 text-white' : 'hover:bg-gray-100'
                  }`}
                >
                  <FileText className="w-5 h-5" />
                  All Ideas
                </button>
                {currentUser?.role === 'NORMAL' && (
                  <button
                    onClick={() => {
                      setActiveView('preferences');
                      setMobileMenuOpen(false);
                    }}
                    className={`w-full text-left px-4 py-3 rounded-lg transition flex items-center gap-2 ${
                      activeView === 'preferences' ? 'bg-indigo-600 text-white' : 'hover:bg-gray-100'
                    }`}
                  >
                    <Award className="w-5 h-5" />
                    My Preferences
                  </button>
                )}
                <button
                  onClick={() => {
                    setActiveView('allocations');
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition flex items-center gap-2 ${
                    activeView === 'allocations' ? 'bg-indigo-600 text-white' : 'hover:bg-gray-100'
                  }`}
                >
                  <Users className="w-5 h-5" />
                  Allocations
                </button>
                <button
                  onClick={() => {
                    setActiveView('top');
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full text-left px-4 py-3 rounded-lg transition flex items-center gap-2 ${
                    activeView === 'top' ? 'bg-indigo-600 text-white' : 'hover:bg-gray-100'
                  }`}
                >
                  <BarChart3 className="w-5 h-5" />
                  Top Ideas
                </button>
                <button
                  onClick={() => {
                    setCurrentUser(null);
                    setActiveView('login');
                    setMobileMenuOpen(false);
                  }}
                  className="lg:hidden w-full text-left px-4 py-3 rounded-lg transition hover:bg-red-50 text-red-600"
                >
                  Logout
                </button>
              </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1">
              {activeView === 'submit' && <SubmitIdeaView />}
              {activeView === 'ideas' && <AllIdeasView />}
              {activeView === 'preferences' && <SubmitPreferencesView />}
              {activeView === 'allocations' && <AllocationView />}
              {activeView === 'top' && <TopIdeasView />}
            </main>
          </div>
        </div>
      </div>
    );
  };

  if (!currentUser) {
    return <LoginView />;
  }

  return <DashboardView />;
};

export default App;